import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
import json

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Expose-Headers": "Access-Control-Allow-Origin",
}

app = FastAPI()

class CodeRequest(BaseModel):
    code: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]

def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout

def analyze_error_with_ai(code: str, traceback_str: str) -> List[int]:
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
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        return data.get("error_lines", [])
    except json.JSONDecodeError:
        return []

@app.get("/")
async def root():
    return JSONResponse(content={"status": "ok"}, headers=CORS_HEADERS)

@app.options("/code-interpreter")
async def options_handler():
    return JSONResponse(content={}, headers=CORS_HEADERS)

@app.post("/code-interpreter")
async def code_interpreter(request: CodeRequest):
    execution_result = execute_python_code(request.code)

    if execution_result["success"]:
        return JSONResponse(
            content={"error": [], "result": execution_result["output"]},
            headers=CORS_HEADERS
        )

    error_lines = analyze_error_with_ai(request.code, execution_result["output"])

    return JSONResponse(
        content={"error": error_lines, "result": execution_result["output"]},
        headers=CORS_HEADERS
    )