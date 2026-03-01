import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
import json

# ─── CORS headers (required by the assignment) ───────────────────────────────
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Expose-Headers": "Access-Control-Allow-Origin",
}

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Access-Control-Allow-Origin"],
)

# ─── Request/Response models ──────────────────────────────────────────────────
class CodeRequest(BaseModel):
    code: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]

# ─── PART 1: Tool function — runs the code ────────────────────────────────────
def execute_python_code(code: str) -> dict:
    """Run Python code, capture output or error traceback."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, {})  # run the code in a clean namespace
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout  # always restore stdout

# ─── PART 2: AI error analysis — asks Groq which line broke ──────────────────
def analyze_error_with_ai(code: str, traceback_str: str) -> List[int]:
    """Send code + traceback to Groq; get back the broken line numbers."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    prompt = f"""You are a Python debugging assistant.
Analyze the code and its error traceback below.
Identify the line number(s) in the ORIGINAL code where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_str}

Respond with ONLY valid JSON in this exact format, nothing else:
{{"error_lines": [3]}}

If multiple lines caused errors, list them all: {{"error_lines": [2, 5]}}"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",   # free Groq model
        messages=[{"role": "user", "content": prompt}],
        temperature=0,            # deterministic answers
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()

    # Parse the JSON response safely
    try:
        data = json.loads(raw)
        return data.get("error_lines", [])
    except json.JSONDecodeError:
        # If AI gave weird output, fall back to empty list
        return []

# ─── PART 3: The actual endpoint ─────────────────────────────────────────────
@app.options("/code-interpreter")
async def options_handler():
    """Handle CORS preflight requests."""
    return JSONResponse(content={}, headers=CORS_HEADERS)

@app.post("/code-interpreter")
async def code_interpreter(request: CodeRequest):
    # Step 1: Run the code
    execution_result = execute_python_code(request.code)

    # Step 2: If it worked, return immediately (no AI needed)
    if execution_result["success"]:
        return JSONResponse(
            content={"error": [], "result": execution_result["output"]},
            headers=CORS_HEADERS
        )

    # Step 3: It failed — ask AI to find the broken line(s)
    error_lines = analyze_error_with_ai(request.code, execution_result["output"])

    return JSONResponse(
        content={
            "error": error_lines,
            "result": execution_result["output"]
        },
        headers=CORS_HEADERS
    )
