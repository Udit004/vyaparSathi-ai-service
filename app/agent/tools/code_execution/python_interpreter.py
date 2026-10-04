"""
app/agent/tools/code_execution/python_interpreter.py
======================================================
Sandboxed Python Execution Tool for Vyapar Sathi AI Agent & Voice Assistant.

Enables the AI to perform complex mathematical calculations, data analysis,
financial forecasts, and custom algorithms safely.
"""

import sys
import io
import math
import json
import datetime
import random
import re
import multiprocessing
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class PythonCodeInput(BaseModel):
    code: str = Field(
        ...,
        description="The Python code block to execute. Use print() to output results or assign the answer to a variable named `result`."
    )


def _worker_exec(code: str, queue: multiprocessing.Queue):
    """Executes Python code in an isolated worker process with strict globals and stdout capture."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    safe_builtins = {
        'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
        'bytearray': bytearray, 'bytes': bytes, 'chr': chr, 'complex': complex,
        'dict': dict, 'divmod': divmod, 'enumerate': enumerate, 'filter': filter,
        'float': float, 'format': format, 'frozenset': frozenset, 'hasattr': hasattr,
        'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
        'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list,
        'map': map, 'max': max, 'min': min, 'next': next, 'oct': oct,
        'ord': ord, 'pow': pow, 'range': range, 'repr': repr,
        'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
        'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
        'zip': zip, 'True': True, 'False': False, 'None': None
    }

    allowed_globals: Dict[str, Any] = {
        '__builtins__': safe_builtins,
        'math': math,
        'json': json,
        'datetime': datetime,
        'random': random,
        're': re,
    }

    def custom_print(*args, **kwargs):
        kwargs['file'] = stdout_buffer
        print(*args, **kwargs)

    safe_builtins['print'] = custom_print

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        local_vars: Dict[str, Any] = {}
        exec(code, allowed_globals, local_vars)

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        out_str = stdout_buffer.getvalue()
        err_str = stderr_buffer.getvalue()
        result_val = local_vars.get('result', None)

        queue.put({
            "success": True,
            "stdout": out_str,
            "stderr": err_str,
            "result": str(result_val) if result_val is not None else None,
            "error": None
        })
    except Exception as exc:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        queue.put({
            "success": False,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "result": None,
            "error": f"{type(exc).__name__}: {str(exc)}"
        })


@tool("execute_python_code", args_schema=PythonCodeInput)
async def execute_python_code(code: str) -> str:
    """Execute Python code for math calculations, data analysis, profit/loss modeling, or algorithms.
    Use print() to show output or set `result = ...` for the return value.
    """
    # Safety sanitization
    forbidden_terms = [
        'import os', 'import sys', 'subprocess', 'open(', 'eval(', 'exec(',
        '__import__', 'shutil', 'socket', 'urllib', 'requests', 'pathlib'
    ]
    for term in forbidden_terms:
        if term in code:
            return f"Security Error: Usage of '{term}' is restricted for safety reasons."

    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_worker_exec, args=(code, queue))
    proc.start()
    proc.join(timeout=5.0)  # 5-second execution timeout

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return "Execution Error: Python code execution timed out (limit: 5 seconds)."

    if not queue.empty():
        res = queue.get()
        if res["success"]:
            output_parts = []
            if res["stdout"]:
                output_parts.append(f"Output:\n{res['stdout'].strip()}")
            if res["result"] is not None:
                output_parts.append(f"Result: {res['result']}")
            if not output_parts:
                output_parts.append("Code executed successfully (no output).")
            return "\n".join(output_parts)
        else:
            return f"Execution Error: {res['error']}\n{res['stderr']}".strip()

    return "Execution Error: Worker process failed to return output."
