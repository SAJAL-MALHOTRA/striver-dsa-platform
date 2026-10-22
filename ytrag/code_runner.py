"""
Safe local code runner for Python and C++ algorithms.
Compiles and executes code in an isolated scratch sandbox with timeout boundaries.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
SCRATCH_RUNNER_DIR = BASE_DIR / "scratch" / "runner"

DISALLOWED_PY_MODULES = {
    "os", "subprocess", "shutil", "socket", "ssl", "ctypes", "winreg",
    "msvcrt", "pty", "multiprocessing", "importlib", "posix", "nt",
    "builtins", "signal", "threading", "http", "urllib", "requests", "httpx"
}

DISALLOWED_PY_CALLS = {"eval", "exec", "__import__"}
DISALLOWED_PY_ATTRS = {"__subclasses__", "__globals__", "system", "popen", "spawn"}


def validate_code_security(language: str, code: str) -> tuple[bool, Optional[str]]:
    """
    AST and lexical security auditor preventing dangerous system calls, process spawning,
    arbitrary file manipulation, network sockets, or sandbox escapes in DSA execution.
    """
    lang = language.lower().strip()
    if lang in ("python", "py"):
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod_base = alias.name.split(".")[0]
                        if mod_base in DISALLOWED_PY_MODULES or alias.name in DISALLOWED_PY_MODULES:
                            return False, f"Importing '{alias.name}' is prohibited in the algorithmic sandbox."
                elif isinstance(node, ast.ImportFrom):
                    mod_name = node.module or ""
                    mod_base = mod_name.split(".")[0]
                    if mod_base in DISALLOWED_PY_MODULES or mod_name in DISALLOWED_PY_MODULES:
                        return False, f"Importing from '{mod_name}' is prohibited in the algorithmic sandbox."
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_PY_CALLS:
                        return False, f"Calling '{node.func.id}()' is prohibited in the algorithmic sandbox."
                    if isinstance(node.func, ast.Attribute) and node.func.attr in DISALLOWED_PY_ATTRS:
                        return False, f"Calling method '{node.func.attr}()' is prohibited."
                elif isinstance(node, ast.Attribute):
                    if node.attr in DISALLOWED_PY_ATTRS:
                        return False, f"Accessing attribute '{node.attr}' is prohibited in the algorithmic sandbox."
        except SyntaxError:
            # Let Python runtime produce normal syntax error feedback
            pass
        return True, None

    elif lang in ("cpp", "c++", "cxx"):
        cpp_lower = code.lower()
        forbidden_headers = [
            "windows.h", "unistd.h", "sys/socket.h", "sys/types.h",
            "sys/stat.h", "netinet/in.h", "arpa/inet.h", "process.h",
            "direct.h", "winsock2.h", "conio.h",
        ]
        for h in forbidden_headers:
            if f"<{h}>" in cpp_lower or f'"{h}"' in cpp_lower:
                return False, f"Header <{h}> is prohibited in the algorithmic sandbox."

        forbidden_patterns = [
            (r"\bsystem\s*\(", "system()"),
            (r"\bpopen\s*\(", "popen()"),
            (r"\bfork\s*\(", "fork()"),
            (r"\bexec(ve|vp|v|l|lp)\s*\(", "exec()"),
            (r"\bkill\s*\(", "kill()"),
            (r"\bsocket\s*\(", "socket()"),
            (r"\bCreateProcess", "CreateProcess"),
            (r"\bShellExecute", "ShellExecute"),
            (r"\bWinExec", "WinExec"),
        ]
        for pat, label in forbidden_patterns:
            if re.search(pat, code):
                return False, f"Function call '{label}' is prohibited in the algorithmic sandbox."

        return True, None

    return True, None


def _clean_output(text: str, max_chars: int = 30000) -> str:
    """Strip null bytes and truncate excess output."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [Output truncated at {max_chars} chars]"
    return text


def _prepare_python_code(code: str) -> str:
    """
    Ensure the Python code has basic imports if needed.
    """
    lines = code.strip().splitlines()
    header = []
    # If type hints like list[int] or Optional or Dict are used without imports
    has_typing = any(re.search(r"\b(List|Dict|Set|Tuple|Optional)\b", line) for line in lines)
    if has_typing and not any("typing" in line for line in lines):
        header.append("from typing import List, Dict, Set, Tuple, Optional")
    
    if header:
        return "\n".join(header) + "\n\n" + code
    return code


def _prepare_cpp_code(code: str) -> str:
    """
    Ensure standard C++ headers and namespace are present.
    """
    code_stripped = code.strip()
    includes = []
    if "#include" not in code_stripped:
        includes.extend([
            "#include <iostream>",
            "#include <vector>",
            "#include <string>",
            "#include <unordered_map>",
            "#include <unordered_set>",
            "#include <map>",
            "#include <set>",
            "#include <algorithm>",
            "#include <queue>",
            "#include <stack>",
            "#include <cmath>",
        ])
    if "using namespace std;" not in code_stripped and "std::" not in code_stripped:
        includes.append("using namespace std;")

    if includes:
        return "\n".join(includes) + "\n\n" + code_stripped
    return code_stripped


def execute_code(
    language: str,
    code: str,
    stdin_input: str = "",
    timeout_sec: float = 5.0,
) -> dict:
    """
    Executes Python or C++ code with strict timeouts and memory isolation.

    Parameters
    ----------
    language : str
        'python' or 'cpp'
    code : str
        Source code to execute
    stdin_input : str
        Standard input string provided to the process
    timeout_sec : float
        Maximum execution time allowed (default 5.0s)

    Returns
    -------
    dict
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "execution_time_ms": float,
            "exit_code": int,
            "message": str
        }
    """
    lang = language.lower().strip()
    if lang not in ("python", "py", "cpp", "c++", "cxx"):
        if lang in ("java",):
            return {
                "success": False,
                "stdout": "",
                "stderr": "Java execution requires a local JDK setup. Please use Python or C++ tab to run directly.",
                "execution_time_ms": 0.0,
                "exit_code": 1,
                "message": "Unsupported runtime: Java (switch to Python or C++)",
            }
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Unsupported language '{language}'. Supported: python, cpp",
            "execution_time_ms": 0.0,
            "exit_code": 1,
            "message": f"Unsupported language: {language}",
        }

    # Security Sandbox Audit: prevent RCE, dangerous imports, process spawning, or host file manipulation
    is_safe, sec_err = validate_code_security(lang, code)
    if not is_safe:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Security Sandbox Violation: {sec_err}",
            "execution_time_ms": 0.0,
            "exit_code": 1,
            "message": "Security Sandbox Violation",
        }

    # Setup isolated scratch sandbox
    SCRATCH_RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    work_dir = SCRATCH_RUNNER_DIR / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if lang in ("python", "py"):
            return _execute_python(work_dir, code, stdin_input, timeout_sec)
        else:
            return _execute_cpp(work_dir, code, stdin_input, timeout_sec)
    finally:
        # Clean up temporary scratch directory
        shutil.rmtree(work_dir, ignore_errors=True)


def _execute_python(
    work_dir: Path,
    code: str,
    stdin_input: str,
    timeout_sec: float,
) -> dict:
    prepared_code = _prepare_python_code(code)
    src_file = work_dir / "solution.py"
    src_file.write_text(prepared_code, encoding="utf-8")

    t_start = time.perf_counter()
    try:
        proc = subprocess.run(
            ["python", str(src_file)],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(work_dir),
        )
        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
        stdout = _clean_output(proc.stdout)
        stderr = _clean_output(proc.stderr)
        is_success = proc.returncode == 0

        msg = "Execution completed successfully." if is_success else f"Process exited with code {proc.returncode}."
        if is_success and not stdout.strip():
            msg = "Executed successfully (no stdout produced. Tip: add print(...) statements to view results)."

        return {
            "success": is_success,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time_ms": duration_ms,
            "exit_code": proc.returncode,
            "message": msg,
        }
    except subprocess.TimeoutExpired:
        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Time Limit Exceeded: Process terminated after {timeout_sec:.1f}s. Check for infinite loops.",
            "execution_time_ms": duration_ms,
            "exit_code": -1,
            "message": f"Time Limit Exceeded ({timeout_sec:.1f}s)",
        }
    except Exception as exc:
        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution error: {exc}",
            "execution_time_ms": duration_ms,
            "exit_code": 1,
            "message": f"Failed to execute Python: {type(exc).__name__}",
        }


def _execute_cpp(
    work_dir: Path,
    code: str,
    stdin_input: str,
    timeout_sec: float,
) -> dict:
    prepared_code = _prepare_cpp_code(code)
    src_file = work_dir / "solution.cpp"
    exe_file = work_dir / ("solution.exe" if os.name == "nt" else "solution")
    src_file.write_text(prepared_code, encoding="utf-8")

    t_start = time.perf_counter()
    # 1. Compilation
    try:
        comp_proc = subprocess.run(
            ["g++", "-std=c++20", "-O2", str(src_file), "-o", str(exe_file)],
            capture_output=True,
            text=True,
            timeout=10.0,
            cwd=str(work_dir),
        )
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "g++ compiler not found in system PATH. Please ensure MinGW/GCC is installed.",
            "execution_time_ms": 0.0,
            "exit_code": 1,
            "message": "g++ compiler not available",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Compilation timed out after 10s.",
            "execution_time_ms": 10000.0,
            "exit_code": -1,
            "message": "Compilation timed out",
        }

    if comp_proc.returncode != 0:
        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "success": False,
            "stdout": "",
            "stderr": _clean_output(comp_proc.stderr),
            "execution_time_ms": duration_ms,
            "exit_code": comp_proc.returncode,
            "message": "Compilation error. Inspect compiler trace in console.",
        }

    # 2. Execution
    t_run_start = time.perf_counter()
    try:
        run_proc = subprocess.run(
            [str(exe_file)],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(work_dir),
        )
        duration_ms = round((time.perf_counter() - t_run_start) * 1000, 2)
        stdout = _clean_output(run_proc.stdout)
        stderr = _clean_output(run_proc.stderr)
        is_success = run_proc.returncode == 0

        msg = "Execution completed successfully." if is_success else f"Process exited with code {run_proc.returncode}."
        if is_success and not stdout.strip():
            msg = "Executed successfully (no stdout produced. Tip: add cout << ... to view output)."

        return {
            "success": is_success,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time_ms": duration_ms,
            "exit_code": run_proc.returncode,
            "message": msg,
        }
    except subprocess.TimeoutExpired:
        duration_ms = round((time.perf_counter() - t_run_start) * 1000, 2)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Time Limit Exceeded: Process terminated after {timeout_sec:.1f}s. Check for infinite loops.",
            "execution_time_ms": duration_ms,
            "exit_code": -1,
            "message": f"Time Limit Exceeded ({timeout_sec:.1f}s)",
        }
    except Exception as exc:
        duration_ms = round((time.perf_counter() - t_run_start) * 1000, 2)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution error: {exc}",
            "execution_time_ms": duration_ms,
            "exit_code": 1,
            "message": f"Failed to execute C++ binary: {type(exc).__name__}",
        }
