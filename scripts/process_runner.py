"""
ARGUS AI Real-Time Subprocess Execution & Streamer.

Executes subprocess commands while continuously streaming stdout and stderr
directly to the active terminal without buffering or suppression.

Usage:
    python scripts/process_runner.py --tag PIP -- python -m pip install ...
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_process_streaming(
    command: list[str] | str,
    cwd: Path | str | None = None,
    env: dict | None = None,
    tag: str | None = None,
    timeout_seconds: int | None = None,
    shell: bool = False,
) -> int:
    """
    Execute a subprocess and stream its output in real time to standard output.
    """
    tag_prefix = f"[{tag}] " if tag else ""
    cmd_str = " ".join(command) if isinstance(command, list) else command

    print(f"[ARGUS PROCESS] Starting: {cmd_str}", flush=True)

    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    # Ensure Python child processes do not buffer output
    proc_env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=shell,
        )

        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                clean_line = line.rstrip("\r\n")
                if clean_line:
                    print(f"{tag_prefix}{clean_line}", flush=True)

        proc.stdout.close()
        return_code = proc.wait(timeout=timeout_seconds)

        status_msg = "SUCCESS" if return_code == 0 else f"FAILED (Exit Code: {return_code})"
        print(f"[ARGUS PROCESS] {status_msg}: {cmd_str}\n", flush=True)
        return return_code

    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"[ARGUS ERROR] Process timed out after {timeout_seconds}s: {cmd_str}", flush=True)
        return 124
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        print(f"[ARGUS ERROR] Execution error ({e}): {cmd_str}", flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGUS Subprocess Streaming Runner")
    parser.add_argument("--tag", default=None, help="Prefix tag for output lines (e.g. PIP)")
    parser.add_argument("--cwd", default=None, help="Working directory")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command and arguments to execute")

    args = parser.parse_args()
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        print("[ARGUS ERROR] No command provided to process_runner.py", flush=True)
        return 1

    return run_process_streaming(
        command=cmd,
        cwd=args.cwd,
        tag=args.tag,
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
