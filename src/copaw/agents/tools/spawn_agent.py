# -*- coding: utf-8 -*-
"""Built-in tool for delegating one-shot tasks to external runners."""

import asyncio
import locale
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from ...config import load_config
from ...config.config import (
    AgentProfileConfig,
    SpawnAgentRunnerConfig,
    load_agent_config,
    resolve_spawn_agent_runners,
)
from ...config.context import get_current_workspace_dir
from ...constant import WORKING_DIR


def _smart_decode(data: bytes) -> str:
    """Decode subprocess output using UTF-8 with locale fallback."""
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        encoding = locale.getpreferredencoding(False) or "utf-8"
        decoded = data.decode(encoding, errors="replace")
    return decoded.strip("\n")


def _kill_process_tree_win32(pid: int) -> None:
    """Kill a process tree on Windows."""
    try:
        subprocess.call(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        pass


def _response_text(text: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=text)])


def _validate_spawn_agent_inputs(
    task_text: str,
    runner_name: str,
    timeout: int,
) -> Optional[str]:
    """Return a user-facing validation error for invalid tool inputs."""
    if not task_text:
        return "Error: task is empty."
    if not runner_name:
        return "Error: agent_type is empty."
    if timeout <= 0:
        return "Error: timeout must be greater than 0."
    return None


def _current_workspace_dir() -> Path:
    return (get_current_workspace_dir() or Path(WORKING_DIR)).expanduser()


def _find_current_agent_id(workspace_dir: Path) -> str:
    """Resolve the active agent ID from the current workspace path."""
    config = load_config()
    target = workspace_dir.expanduser().resolve()

    for agent_id, profile in config.agents.profiles.items():
        candidate = Path(profile.workspace_dir).expanduser().resolve()
        if candidate == target:
            return agent_id

    raise ValueError(
        "Unable to resolve the current agent from the workspace directory.",
    )


def _resolve_current_agent_config() -> tuple[str, AgentProfileConfig, Path]:
    workspace_dir = _current_workspace_dir()
    agent_id = _find_current_agent_id(workspace_dir)
    return agent_id, load_agent_config(agent_id), workspace_dir


def _resolve_runner(
    agent_config: AgentProfileConfig,
    agent_type: str,
) -> SpawnAgentRunnerConfig:
    available_runners = resolve_spawn_agent_runners(agent_config.spawn_agent)
    runner = available_runners.get(agent_type)
    if runner is None:
        known = sorted(available_runners)
        known_text = ", ".join(known) if known else "none"
        raise ValueError(
            f"Unknown spawn_agent runner '{agent_type}'. "
            f"Configured runners: {known_text}.",
        )

    if not runner.enabled:
        raise ValueError(
            f"spawn_agent runner '{agent_type}' is disabled.",
        )

    if not runner.command.strip():
        raise ValueError(
            f"spawn_agent runner '{agent_type}' is missing command.",
        )

    return runner


def _resolve_runner_args(args: list[str], task: str) -> list[str]:
    resolved_args: list[str] = []
    saw_placeholder = False

    for arg in args:
        if "{task}" in arg:
            resolved_args.append(arg.replace("{task}", task))
            saw_placeholder = True
        else:
            resolved_args.append(arg)

    if not saw_placeholder:
        resolved_args.append(task)

    return resolved_args


def _resolve_execution_cwd(
    cwd: Optional[str],
    runner: SpawnAgentRunnerConfig,
    workspace_dir: Path,
) -> Path:
    raw_cwd = (cwd or "").strip() or runner.cwd.strip()
    if not raw_cwd:
        return workspace_dir

    candidate = Path(raw_cwd).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_dir / candidate
    return candidate.resolve()


def _resolve_execution_env(runner: SpawnAgentRunnerConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(runner.env)

    python_bin_dir = str(Path(sys.executable).parent)
    current_path = env.get("PATH", "")
    env["PATH"] = (
        python_bin_dir + os.pathsep + current_path
        if current_path
        else python_bin_dir
    )
    return env


def _format_command(command: str, args: list[str]) -> str:
    return " ".join([command, *args]).strip()


def _detect_runner_guidance(
    agent_type: str,
    command: str,
    stderr_text: str,
    env: dict[str, str],
) -> Optional[str]:
    """Translate known runner auth/config failures into actionable guidance."""
    normalized_agent = agent_type.strip().lower()
    normalized_command = command.strip().lower()
    normalized_stderr = stderr_text.lower()

    is_qwen_runner = normalized_agent == "qwen" or (
        normalized_command == "qwen"
    )
    if not is_qwen_runner:
        return None

    if (
        "no auth type is selected" in normalized_stderr
        or "please configure an auth type" in normalized_stderr
    ):
        return (
            "qwen runner is not authenticated. "
            "Please run `qwen auth` locally and complete login first."
        )

    anthropic_override = any(
        env.get(key, "").strip()
        for key in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_MODEL",
        )
    )
    if (
        "request body format invalid" in normalized_stderr
        and anthropic_override
    ):
        return (
            "qwen runner is using ANTHROPIC_* environment overrides that are "
            "not compatible with tool-based delegation. Please configure "
            "`qwen auth` officially, or clear ANTHROPIC_API_KEY / "
            "ANTHROPIC_BASE_URL / ANTHROPIC_MODEL before using qwen."
        )

    return None


def _detect_runner_launch_guidance(
    agent_type: str,
    command: str,
    exc: Exception,
) -> Optional[str]:
    """Translate launch-time errors into install guidance."""
    if not isinstance(exc, FileNotFoundError):
        return None

    normalized_agent = agent_type.strip().lower()
    normalized_command = command.strip()

    if normalized_agent == "opencode" or normalized_command == "opencode":
        return (
            "opencode runner is not installed or not on PATH. "
            "Please install OpenCode CLI and ensure `opencode` is available "
            "in your shell."
        )

    if normalized_agent == "qwen" or normalized_command == "qwen":
        return (
            "qwen runner is not installed or not on PATH. "
            "Please install Qwen CLI and ensure `qwen` is available "
            "in your shell."
        )

    if normalized_agent == "gemini":
        return (
            "gemini runner could not be launched. "
            "Please ensure `npx` is available and can download "
            "`@google/gemini-cli`."
        )

    return None


def _build_result_text(
    agent_type: str,
    command: str,
    args: list[str],
    cwd: Path,
    returncode: int,
    stdout_text: str,
    stderr_text: str,
) -> str:
    header = [
        f"spawn_agent runner: {agent_type}",
        f"working directory: {cwd}",
        f"command: {_format_command(command, args)}",
    ]

    if returncode == 0:
        result = ["spawn_agent completed successfully."]
        if stdout_text:
            result.append(f"[stdout]\n{stdout_text}")
        else:
            result.append("[stdout]\n(no output)")
        if stderr_text:
            result.append(f"[stderr]\n{stderr_text}")
    else:
        result = [f"spawn_agent failed with exit code {returncode}."]
        if stdout_text:
            result.append(f"[stdout]\n{stdout_text}")
        if stderr_text:
            result.append(f"[stderr]\n{stderr_text}")

    return "\n".join([*header, "", *result])


async def _run_process(
    command: str,
    args: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        command,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        env=env,
        start_new_session=(sys.platform != "win32"),
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        if sys.platform == "win32":
            _kill_process_tree_win32(proc.pid)
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError, AttributeError):
                proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=2)
            except (ProcessLookupError, OSError, asyncio.TimeoutError):
                pass
        return -1, "", f"spawn_agent timed out after {timeout} seconds."

    return proc.returncode or 0, _smart_decode(stdout), _smart_decode(stderr)


async def spawn_agent(
    task: str,
    agent_type: str,
    cwd: Optional[str] = None,
    timeout: int = 900,
) -> ToolResponse:
    """Delegate a one-shot task to a configured external agent runner."""
    task_text = (task or "").strip()
    runner_name = (agent_type or "").strip()
    validation_error = _validate_spawn_agent_inputs(
        task_text,
        runner_name,
        timeout,
    )
    if validation_error is not None:
        return _response_text(validation_error)

    try:
        _, agent_config, workspace_dir = _resolve_current_agent_config()
        runner = _resolve_runner(agent_config, runner_name)
        args = _resolve_runner_args(runner.args, task_text)
        execution_cwd = _resolve_execution_cwd(cwd, runner, workspace_dir)
        env = _resolve_execution_env(runner)
        try:
            returncode, stdout_text, stderr_text = await _run_process(
                runner.command,
                args,
                execution_cwd,
                env,
                timeout,
            )
        except Exception as exc:
            guidance = _detect_runner_launch_guidance(
                runner_name,
                runner.command,
                exc,
            )
            if guidance is not None:
                return _response_text(guidance)
            raise
        guidance = _detect_runner_guidance(
            runner_name,
            runner.command,
            stderr_text,
            env,
        )
        if guidance is not None:
            return _response_text(guidance)
        return _response_text(
            _build_result_text(
                runner_name,
                runner.command,
                args,
                execution_cwd,
                returncode,
                stdout_text,
                stderr_text,
            ),
        )
    except Exception as exc:
        return _response_text(f"Error: {exc}")
