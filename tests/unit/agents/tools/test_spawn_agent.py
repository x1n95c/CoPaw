# -*- coding: utf-8 -*-
"""Tests for the spawn_agent tool."""

# pylint: disable=redefined-outer-name
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from copaw.config.config import (
    AgentProfileConfig,
    AgentProfileRef,
    AgentsConfig,
    Config,
    SpawnAgentConfig,
    SpawnAgentRunnerConfig,
    save_agent_config,
)
from copaw.config.context import set_current_workspace_dir
from copaw.config.utils import get_config_path
from copaw.security.tool_guard.guardians.rule_guardian import (
    RuleBasedToolGuardian,
)
from copaw.security.tool_guard.guardians.rule_guardian import (
    load_rules_from_directory,
)
from copaw.security.tool_guard.guardians import rule_guardian as rule_guardian_module
from copaw.security.tool_guard.utils import resolve_guarded_tools

spawn_agent_module = importlib.import_module(
    "copaw.agents.tools.spawn_agent",
)


@pytest.fixture
def spawn_agent_workspace(tmp_path, monkeypatch):
    """Create an isolated workspace and agent config for spawn_agent."""
    workspace_dir = tmp_path / "workspaces" / "test_agent"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("copaw.constant.WORKING_DIR", tmp_path)
    monkeypatch.setattr("copaw.config.utils.WORKING_DIR", tmp_path)
    monkeypatch.setattr("copaw.config.config.WORKING_DIR", tmp_path)

    root_config = Config(
        agents=AgentsConfig(
            active_agent="test_agent",
            profiles={
                "test_agent": AgentProfileRef(
                    id="test_agent",
                    workspace_dir=str(workspace_dir),
                ),
            },
        ),
    )

    config_path = Path(get_config_path())
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as file_obj:
        json.dump(root_config.model_dump(exclude_none=True), file_obj)

    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(workspace_dir),
    )
    save_agent_config("test_agent", agent_config)

    set_current_workspace_dir(workspace_dir)
    yield workspace_dir
    set_current_workspace_dir(None)


def _set_runner(
    workspace_id: str,
    agent_config: AgentProfileConfig,
    runner_name: str,
    runner: SpawnAgentRunnerConfig,
) -> None:
    agent_config.spawn_agent = SpawnAgentConfig(
        runners={runner_name: runner},
    )
    save_agent_config(workspace_id, agent_config)


def _response_text(response) -> str:
    return response.content[0]["text"]


@pytest.mark.anyio
async def test_spawn_agent_executes_runner_with_placeholder(
    spawn_agent_workspace,
):
    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    _set_runner(
        "test_agent",
        agent_config,
        "codex",
        SpawnAgentRunnerConfig(
            enabled=True,
            command=sys.executable,
            args=["-c", "import sys; print(sys.argv[1])", "{task}"],
        ),
    )

    response = await spawn_agent_module.spawn_agent(
        "review src/copaw",
        "codex",
    )
    text = _response_text(response)

    assert "spawn_agent completed successfully." in text
    assert "[stdout]\nreview src/copaw" in text


@pytest.mark.anyio
async def test_spawn_agent_appends_task_when_placeholder_missing(
    spawn_agent_workspace,
):
    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    _set_runner(
        "test_agent",
        agent_config,
        "codex",
        SpawnAgentRunnerConfig(
            enabled=True,
            command=sys.executable,
            args=["-c", "import sys; print(sys.argv[1])"],
        ),
    )

    response = await spawn_agent_module.spawn_agent("summarize tests", "codex")
    assert "[stdout]\nsummarize tests" in _response_text(response)


@pytest.mark.anyio
async def test_spawn_agent_uses_runner_cwd_and_call_override(
    spawn_agent_workspace,
):
    runner_dir = spawn_agent_workspace / "runner"
    override_dir = spawn_agent_workspace / "override"
    runner_dir.mkdir()
    override_dir.mkdir()

    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    _set_runner(
        "test_agent",
        agent_config,
        "codex",
        SpawnAgentRunnerConfig(
            enabled=True,
            command=sys.executable,
            args=["-c", "import os; print(os.getcwd())"],
            cwd="runner",
        ),
    )

    response = await spawn_agent_module.spawn_agent("print cwd", "codex")
    assert f"[stdout]\n{runner_dir.resolve()}" in _response_text(response)

    response = await spawn_agent_module.spawn_agent(
        "print cwd",
        "codex",
        cwd="override",
    )
    assert f"[stdout]\n{override_dir.resolve()}" in _response_text(response)


@pytest.mark.anyio
async def test_spawn_agent_errors_for_missing_runner(
    spawn_agent_workspace,
):
    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
        spawn_agent=SpawnAgentConfig(runners={}),
    )
    save_agent_config("test_agent", agent_config)

    response = await spawn_agent_module.spawn_agent("review", "missing")
    assert "Unknown spawn_agent runner 'missing'" in _response_text(response)


@pytest.mark.anyio
async def test_spawn_agent_uses_builtin_runner_presets_by_default(
    spawn_agent_workspace,
    monkeypatch,
):
    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    save_agent_config("test_agent", agent_config)

    monkeypatch.setattr(
        spawn_agent_module,
        "_run_process",
        AsyncMock(return_value=(0, "built-in runner ok", "")),
    )

    response = await spawn_agent_module.spawn_agent("review", "qwen")
    text = _response_text(response)

    assert "spawn_agent completed successfully." in text
    assert "command: qwen --approval-mode yolo review" in text


@pytest.mark.anyio
async def test_spawn_agent_reports_missing_builtin_command(
    spawn_agent_workspace,
    monkeypatch,
):
    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    save_agent_config("test_agent", agent_config)

    async def _raise_missing_command(*_args, **_kwargs):
        raise FileNotFoundError("qwen")

    monkeypatch.setattr(
        spawn_agent_module,
        "_run_process",
        _raise_missing_command,
    )

    response = await spawn_agent_module.spawn_agent("review", "qwen")
    text = _response_text(response)

    assert "qwen runner is not installed or not on PATH" in text


@pytest.mark.anyio
async def test_spawn_agent_errors_for_disabled_runner(
    spawn_agent_workspace,
):
    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    _set_runner(
        "test_agent",
        agent_config,
        "codex",
        SpawnAgentRunnerConfig(
            enabled=False,
            command=sys.executable,
        ),
    )

    response = await spawn_agent_module.spawn_agent("review", "codex")
    assert "runner 'codex' is disabled" in _response_text(response)


@pytest.mark.anyio
async def test_spawn_agent_errors_for_missing_command(
    spawn_agent_workspace,
):
    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    _set_runner(
        "test_agent",
        agent_config,
        "codex",
        SpawnAgentRunnerConfig(
            enabled=True,
            command="",
        ),
    )

    response = await spawn_agent_module.spawn_agent("review", "codex")
    assert "runner 'codex' is missing command" in _response_text(response)


@pytest.mark.anyio
async def test_spawn_agent_reports_timeout(
    spawn_agent_workspace,
):
    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    _set_runner(
        "test_agent",
        agent_config,
        "codex",
        SpawnAgentRunnerConfig(
            enabled=True,
            command=sys.executable,
            args=["-c", "import time; time.sleep(2)"],
        ),
    )

    response = await spawn_agent_module.spawn_agent(
        "sleep",
        "codex",
        timeout=1,
    )
    text = _response_text(response)
    assert "spawn_agent failed with exit code -1." in text
    assert "spawn_agent timed out after 1 seconds." in text


@pytest.mark.anyio
async def test_spawn_agent_reports_nonzero_exit(
    spawn_agent_workspace,
):
    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    _set_runner(
        "test_agent",
        agent_config,
        "codex",
        SpawnAgentRunnerConfig(
            enabled=True,
            command=sys.executable,
            args=[
                "-c",
                (
                    "import sys; "
                    "print('runner stdout'); "
                    "sys.stderr.write('runner stderr\\n'); "
                    "sys.exit(3)"
                ),
            ],
        ),
    )

    response = await spawn_agent_module.spawn_agent("review", "codex")
    text = _response_text(response)

    assert "spawn_agent failed with exit code 3." in text
    assert "[stdout]\nrunner stdout" in text
    assert "[stderr]\nrunner stderr" in text


@pytest.mark.anyio
async def test_spawn_agent_reports_qwen_auth_required(
    spawn_agent_workspace,
):
    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    _set_runner(
        "test_agent",
        agent_config,
        "qwen",
        SpawnAgentRunnerConfig(
            enabled=True,
            command=sys.executable,
            args=[
                "-c",
                (
                    "import sys; "
                    "sys.stderr.write("
                    "'No auth type is selected. Please configure an auth "
                    "type first.\\n'"
                    "); "
                    "sys.exit(1)"
                ),
            ],
        ),
    )

    response = await spawn_agent_module.spawn_agent("review", "qwen")
    text = _response_text(response)

    assert "qwen runner is not authenticated" in text
    assert "qwen auth" in text


@pytest.mark.anyio
async def test_spawn_agent_reports_qwen_anthropic_override_issue(
    spawn_agent_workspace,
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv(
        "ANTHROPIC_BASE_URL",
        "https://coding.dashscope.aliyuncs.com/apps/anthropic",
    )
    monkeypatch.setenv("ANTHROPIC_MODEL", "kimi-k2.5")

    agent_config = AgentProfileConfig(
        id="test_agent",
        name="Test Agent",
        workspace_dir=str(spawn_agent_workspace),
    )
    _set_runner(
        "test_agent",
        agent_config,
        "qwen",
        SpawnAgentRunnerConfig(
            enabled=True,
            command=sys.executable,
            args=[
                "-c",
                (
                    "import sys; "
                    "sys.stderr.write("
                    '\'[API Error: 400 {\\"error\\":{\\"message\\":'
                    '\\"Request body format invalid\\"}}]\\n\''
                    "); "
                    "sys.exit(1)"
                ),
            ],
        ),
    )

    response = await spawn_agent_module.spawn_agent("review", "qwen")
    text = _response_text(response)

    assert "qwen runner is using ANTHROPIC_* environment overrides" in text
    assert "clear ANTHROPIC_API_KEY" in text
    assert "qwen auth" in text


def test_spawn_agent_is_guarded_by_default(monkeypatch):
    guarded = resolve_guarded_tools()
    assert guarded is not None
    assert "spawn_agent" in guarded

    rules_dir = Path("src/copaw/security/tool_guard/rules")
    builtin_rules = load_rules_from_directory(
        rules_dir,
        rule_files=["spawn_agent_commands.yaml"],
    )
    monkeypatch.setattr(
        rule_guardian_module,
        "_load_config_rules",
        lambda: ([], set()),
    )
    guardian = RuleBasedToolGuardian(extra_rules=builtin_rules)
    findings = guardian.guard(
        "spawn_agent",
        {"task": "Review the repository.", "agent_type": "codex"},
    )

    assert findings
    assert findings[0].rule_id == "TOOL_SPAWN_AGENT_EXTERNAL_DELEGATION"
