# -*- coding: utf-8 -*-
"""Shared data types for ACP runtime integration.

This module defines the core data structures used throughout the ACP
integration, including event types, session management, and configuration
parsing from natural language or structured requests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------

ACPEventType = Literal[
    "assistant_chunk",
    "thought_chunk",
    "tool_start",
    "tool_update",
    "tool_end",
    "plan_update",
    "commands_update",
    "usage_update",
    "permission_request",
    "permission_resolved",
    "run_finished",
    "error",
]


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Configuration Types
# ---------------------------------------------------------------------------

@dataclass
class ExternalAgentConfig:
    """External ACP agent request config derived from request extras.

    This configuration can come from:
    - Tool call parameters (spawn_agent)
    - Natural language parsing (e.g., "用 opencode 帮我分析代码")
    - Request metadata (biz_params, model_extra)

    Attributes:
        enabled: Whether the external agent is enabled for this request.
        harness: The runner name (e.g., "opencode", "qwen", "gemini").
        keep_session: Whether to keep the session alive for subsequent calls.
        cwd: Working directory for the external agent.
        existing_session_id: ID of an existing session to resume.
        prompt: The task prompt to send to the external agent.
        keep_session_specified: Whether keep_session was explicitly set.
        preapproved: Whether the request has been pre-approved by user.
    """

    enabled: bool
    harness: str
    keep_session: bool = False
    cwd: str | None = None
    existing_session_id: str | None = None
    prompt: str | None = None
    keep_session_specified: bool = False
    preapproved: bool = False


# ---------------------------------------------------------------------------
# Session Types
# ---------------------------------------------------------------------------

@dataclass
class AcpEvent:
    """Internal ACP event emitted by runtime and consumed by handlers.

    Events are generated during ACP protocol communication and represent
    various stages of agent execution (streaming text, tool calls, etc.).

    Attributes:
        type: The event type (e.g., "assistant_chunk", "tool_start").
        chat_id: The chat session identifier.
        session_id: The ACP session identifier (may be None for ephemeral runs).
        payload: Event-specific data.
    """

    type: ACPEventType
    chat_id: str
    session_id: str | None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuspendedPermission:
    """Pending permission request waiting for user decision.

    When the ACP runtime suspends execution pending a permission decision,
    this object carries all information needed to present the choice to the
    user and resume execution after a decision is made.

    Attributes:
        request_id: JSON-RPC request ID to reply to.
        payload: The raw permission request payload.
        options: Available options for the user.
        harness: The harness requesting permission.
        tool_name: The tool requesting permission.
        tool_kind: The kind of operation (write, execute, etc.).
        target: The target of the operation (path, command, etc.).
    """

    request_id: Any
    payload: dict[str, Any]
    options: list[dict[str, Any]]
    harness: str
    tool_name: str
    tool_kind: str
    target: str | None = None

    def format_chat_message(self) -> str:
        """Format as a user-facing chat message with selectable options."""
        target_line = f"\n- Target: `{self.target}`" if self.target else ""
        options_lines = "\n".join(
            f"  - **{opt.get('title', opt.get('optionId', 'Option'))}** "
            f"(optionId: `{opt.get('optionId', opt.get('id', 'unknown'))}`)"
            for opt in self.options
        )
        return (
            f"🔐 **External Agent Permission Request / 外部 Agent 权限请求**\n\n"
            f"- Harness: `{self.harness}`\n"
            f"- Tool: `{self.tool_name}` (kind: `{self.tool_kind}`)"
            f"{target_line}\n\n"
            f"Options / 可选操作:\n{options_lines}\n\n"
            f"Please reply to allow or deny. "
            f"The model will call spawn_agent with your decision.\n"
            f"请回复是否批准，模型将根据您的回复调用 spawn_agent 传递决定。"
        )


@dataclass
class ACPConversationSession:
    """Runtime state for one chat-bound ACP conversation.

    This represents an active ACP session that can be persisted and resumed
    across multiple tool calls.

    Attributes:
        chat_id: The chat session identifier.
        harness: The runner name (e.g., "opencode", "qwen").
        acp_session_id: The ACP protocol session ID.
        cwd: Working directory for this session.
        keep_session: Whether to persist this session.
        capabilities: Runner capabilities from initialize handshake.
        active_run_id: ID of currently running task (if any).
        updated_at: Last update timestamp.
        runtime: Reference to the ACPRuntime instance (not persisted).
        suspended_permission: Pending permission if turn is suspended.
    """

    chat_id: str
    harness: str
    acp_session_id: str
    cwd: str
    keep_session: bool
    capabilities: dict[str, Any] = field(default_factory=dict)
    active_run_id: str | None = None
    updated_at: datetime = field(default_factory=utc_now)
    runtime: Any | None = None
    suspended_permission: SuspendedPermission | None = None


@dataclass
class ACPRunResult:
    """Summary returned after one ACP turn completes.

    Attributes:
        harness: The runner name that was used.
        session_id: The ACP session ID (for potential resume).
        keep_session: Whether the session was kept alive.
        cwd: Working directory that was used.
        suspended_permission: Set when turn is suspended waiting for user
            permission decision. Caller should present the options to the
            user and resume via service.resume_permission().
    """

    harness: str
    session_id: str | None
    keep_session: bool
    cwd: str
    suspended_permission: SuspendedPermission | None = None


# ---------------------------------------------------------------------------
# Harness Name Normalization
# ---------------------------------------------------------------------------

def normalize_harness_name(raw: str | None) -> str:
    """Normalize external-agent harness names from UI or request payloads.

    Handles common aliases and variations:
    - "qwen-code", "qwen code", "qwencode" -> "qwen"
    - "open-code", "open code" -> "opencode"

    Args:
        raw: The raw harness name string.

    Returns:
        Normalized harness name (lowercase, standardized).
    """
    name = (raw or "").strip().lower()
    aliases = {
        "qwen-code": "qwen",
        "qwen code": "qwen",
        "qwencode": "qwen",
        "open-code": "opencode",
        "open code": "opencode",
    }
    return aliases.get(name, name)


# ---------------------------------------------------------------------------
# Text Parsing Utilities
# ---------------------------------------------------------------------------

def _strip_quotes(raw: str | None) -> str | None:
    """Strip surrounding quotes from a string value."""
    if raw is None:
        return None
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value.strip() or None


def _looks_like_path(raw: str | None) -> bool:
    """Check if a string looks like a file path."""
    value = (raw or "").strip()
    return any(token in value for token in ("/", "\\", "~", ".", ":"))


def _pop_option_value(
    text: str,
    option_names: tuple[str, ...],
) -> tuple[str | None, str]:
    """Extract an option value from text (e.g., --cwd /path)."""
    escaped = "|".join(re.escape(name) for name in option_names)
    pattern = re.compile(
        rf"(?<!\S)(?:{escaped})(?:=|\s+)(\"[^\"]+\"|'[^']+'|\S+)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None, text
    value = _strip_quotes(match.group(1))
    updated = (text[: match.start()] + " " + text[match.end() :]).strip()
    return value, re.sub(r"\s{2,}", " ", updated)


def _pop_flag(text: str, flag_names: tuple[str, ...]) -> tuple[bool, str]:
    """Extract a boolean flag from text (e.g., --keep-session)."""
    escaped = "|".join(re.escape(name) for name in flag_names)
    pattern = re.compile(rf"(?<!\S)(?:{escaped})(?!\S)", re.IGNORECASE)
    match = pattern.search(text)
    if match is None:
        return False, text
    updated = (text[: match.start()] + " " + text[match.end() :]).strip()
    return True, re.sub(r"\s{2,}", " ", updated)


def _pop_leading_harness(text: str) -> tuple[str | None, str]:
    """Extract leading harness name from text."""
    match = re.match(
        (
            r"^(opencode|open(?:\s|-)?code|qwen(?:\s*code|-code)?|qwencode)"
            r"\b\s*(.*)$"
        ),
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None, text
    harness = normalize_harness_name(match.group(1).replace("  ", " "))
    return harness, match.group(2).strip()


def _pop_unknown_harness_token(text: str) -> tuple[str | None, str]:
    """Capture harness token even when not a built-in name."""
    match = re.match(
        r"^(?P<harness>[A-Za-z0-9][A-Za-z0-9_-]*)\b\s*(?P<rest>.*)$",
        text,
    )
    if match is None:
        return None, text
    return (
        normalize_harness_name(match.group("harness")),
        match.group("rest").strip(),
    )


def _normalize_prompt(text: str) -> str:
    """Clean up and normalize a prompt string."""
    cleaned = re.sub(r"\s{2,}", " ", text).strip(" \t\r\n,，:：;；。.、")
    return cleaned or "请帮我处理"


# Path extraction patterns
_PATH_TOKEN_PATTERN = re.compile(
    r"(?P<quoted>\"(?:[^\"\\]|\\.)+\"|'(?:[^'\\]|\\.)+')|"
    r"(?P<bare>(?:~|\.{1,2}|/|[A-Za-z]:[\\/])\S*)",
)


def _clean_path_token(raw: str) -> str | None:
    """Clean a path token by stripping quotes and trailing punctuation."""
    value = _strip_quotes(raw)
    if not value:
        return None
    return value.rstrip("，。,.:：;；!！?？)]}>'\"")


def _path_flavor(path: str) -> type[PurePath]:
    """Determine path flavor (Windows vs POSIX) for a given path."""
    if re.match(r"^[A-Za-z]:[\\/]", path):
        return PureWindowsPath
    return PurePosixPath


def _looks_like_file_path(path: str) -> bool:
    """Check if path looks like a file (has extension)."""
    if path.endswith(("/", "\\")):
        return False
    pure_path = _path_flavor(path)(path)
    name = pure_path.name
    return "." in name and name not in {".", ".."}


def _relative_display_path(path: str, cwd: str) -> str:
    """Get relative display path for a file."""
    flavor = _path_flavor(path)
    pure_path = flavor(path)
    pure_cwd = flavor(cwd)
    try:
        relative = pure_path.relative_to(pure_cwd)
    except ValueError:
        return pure_path.name or str(pure_path)
    return str(relative) or pure_path.name or str(pure_path)


def _infer_cwd_from_prompt(
    prompt: str,
) -> tuple[str | None, str]:
    """Infer working directory from prompt text."""
    for match in _PATH_TOKEN_PATTERN.finditer(prompt):
        token = match.group("quoted") or match.group("bare")
        candidate = _clean_path_token(token)
        if not candidate or not _looks_like_path(candidate):
            continue

        if _looks_like_file_path(candidate):
            flavor = _path_flavor(candidate)
            pure_path = flavor(candidate)
            parent = str(pure_path.parent)
            if not parent or parent == ".":
                continue
            display_path = _relative_display_path(candidate, parent)
            updated = (
                prompt[: match.start()]
                + prompt[match.start() : match.end()].replace(
                    token,
                    display_path,
                    1,
                )
                + prompt[match.end() :]
            )
            return parent, re.sub(r"\s{2,}", " ", updated).strip()

        return candidate.rstrip("/\\") or candidate, prompt

    return None, prompt


# Leading text patterns for natural language parsing
_LEADING_CONTROL_NOISE = r"(?:[\s,，:：;；]|(?:and|then|also|并且|并|然后|再|同时)\s+)*"
_LEADING_COURTESY = (
    r"(?:(?:请|请你|请帮我|帮我|麻烦(?:你)?|劳驾)\s*|"
    r"(?:please|can you|could you|would you|help me)\s+)*"
)


def _pop_leading_match(
    text: str,
    pattern: str,
) -> tuple[re.Match[str] | None, str]:
    """Match leading pattern with courtesy prefix handling."""
    match = re.match(
        rf"^{_LEADING_CONTROL_NOISE}{_LEADING_COURTESY}{pattern}$",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None, text
    rest = match.groupdict().get("rest", "")
    return match, re.sub(r"\s{2,}", " ", rest).strip()


# ---------------------------------------------------------------------------
# Main Parsing Functions
# ---------------------------------------------------------------------------

def parse_external_agent_text(  # pylint: disable=too-many-branches,too-many-statements
    raw: str | None,
) -> ExternalAgentConfig | None:
    """Parse ACP intent from command-style or natural-language text.

    Supports multiple input formats:
    - Command style: "/acp opencode --cwd /path task description"
    - Natural language: "用 opencode 帮我分析代码"
    - English: "use opencode to analyze the code"

    Args:
        raw: The input text to parse.

    Returns:
        ExternalAgentConfig if ACP intent detected, None otherwise.
    """
    text = (raw or "").strip()
    if not text:
        return None

    harness: str | None = None
    working = text

    # Handle /acp prefix
    if re.match(r"^/acp\b", text, re.IGNORECASE):
        working = re.sub(
            r"^/acp\b",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        harness, working = _pop_leading_harness(working)
        if harness is None:
            harness, working = _pop_option_value(
                working,
                ("--harness", "--agent"),
            )
            if harness is not None:
                harness = normalize_harness_name(harness)
        if harness is None:
            harness, working = _pop_unknown_harness_token(working)
        if not harness:
            return None
    else:
        # Try slash command format: /opencode task
        slash_match = re.match(
            (
                r"^/"
                r"(opencode|open(?:\s|-)?code|"
                r"qwen(?:\s*code|-code)?|qwencode)"
                r"\b\s*(.*)$"
            ),
            text,
            re.IGNORECASE,
        )
        # Try --harness format
        cli_match = re.match(
            (
                r"^(?:--harness)(?:=|\s+)"
                r"(opencode|open(?:\s|-)?code|qwen(?:\s*code|-code)?|qwencode)"
                r"\b\s*(.*)$"
            ),
            text,
            re.IGNORECASE,
        )
        # Try Chinese natural language
        zh_match = re.match(
            (
                r"^(?:(?:请|请你|请帮我|帮我|麻烦(?:你)?|劳驾)\s*)?"
                r"(?:用|使用|让|通过|调用)\s+"
                r"(opencode|open(?:\s|-)?code|qwen(?:\s*code|-code)?|qwencode)"
                r"\b(?:\s*(?:来|去|帮忙|帮助))?\s*(.*)$"
            ),
            text,
            re.IGNORECASE,
        )
        # Try English natural language
        en_match = re.match(
            (
                r"^(?:(?:please|can you|could you|would you|help me)\s+)?"
                r"(?:use|with|via|call)\s+"
                r"(opencode|open(?:\s|-)?code|qwen(?:\s*code|-code)?|qwencode)"
                r"\b(?:\s+to)?\s*(.*)$"
            ),
            text,
            re.IGNORECASE,
        )
        match = slash_match or cli_match or zh_match or en_match
        if match is None:
            return None
        harness = normalize_harness_name(match.group(1).replace("  ", " "))
        working = match.group(2).strip()

    # Parse options
    keep_session = False
    keep_session_specified = False

    # --keep-session flag
    keep_flag, working = _pop_flag(working, ("--keep-session", "--session"))
    if keep_flag:
        keep_session = True
        keep_session_specified = True

    # --session-id option
    session_id, working = _pop_option_value(
        working,
        ("--session-id", "--resume-session", "--load-session"),
    )

    # Natural language session reference
    if session_id is None:
        natural_session, working = _pop_leading_match(
            working,
            (
                r"(?:继续|复用|加载|use|reuse|load|continue with)\s*"
                r"(?:session|会话)\s+"
                r"(?P<value>\"[^\"]+\"|'[^']+'|\S+)"
                rf"{_LEADING_CONTROL_NOISE}(?P<rest>.*)"
            ),
        )
        if natural_session is not None:
            session_id = _strip_quotes(natural_session.group("value"))

    # --cwd option
    cwd, working = _pop_option_value(
        working,
        ("--cwd", "--workdir", "--working-dir", "--work-path"),
    )

    # Natural language cwd
    if cwd is None:
        explicit_cwd, working = _pop_leading_match(
            working,
            (
                r"(?:工作路径|工作目录|workdir|cwd)\s*(?:是|为|=|:|：)?\s*"
                r"(?P<value>\"[^\"]+\"|'[^']+'|\S+)"
                rf"{_LEADING_CONTROL_NOISE}(?P<rest>.*)"
            ),
        )
        if explicit_cwd is not None:
            cwd = _strip_quotes(explicit_cwd.group("value"))

    if cwd is None:
        natural_cwd, working = _pop_leading_match(
            working,
            (
                r"在\s+(?P<value>\"[^\"]+\"|'[^']+'|\S+)\s+"
                r"(?:下|目录下|工作目录下)"
                rf"{_LEADING_CONTROL_NOISE}(?P<rest>.*)"
            ),
        )
        candidate = (
            _strip_quotes(natural_cwd.group("value"))
            if natural_cwd is not None
            else None
        )
        if natural_cwd is not None and _looks_like_path(candidate):
            cwd = candidate

    # Keep session phrases
    keep_phrase, working = _pop_leading_match(
        working,
        (
            r"(?:保持会话|保留会话|keep(?:\s+the)?\s+session)"
            rf"\b{_LEADING_CONTROL_NOISE}(?P<rest>.*)"
        ),
    )
    if keep_phrase is not None:
        keep_session = True
        keep_session_specified = True

    # Current session reference
    current_session_phrase, working = _pop_leading_match(
        working,
        (
            r"(?:(?:(?:使用|复用|继续用|沿用|在|用)\s*|"
            r"(?:use|reuse|continue with)\s+))?"
            r"(?:(?:之前的|上一个|上次的|刚才的|当前的?|现在的?)\s*"
            r"(?:acp\s*)?(?:session|会话)|"
            r"(?:the\s+)?(?:previous|last|current)\s+(?:acp\s+)?session)"
            r"(?:\s*用)?"
            rf"{_LEADING_CONTROL_NOISE}(?P<rest>.*)"
        ),
    )
    if current_session_phrase is not None:
        keep_session = True
        keep_session_specified = True

    # Session ID implies keep_session
    if session_id:
        keep_session = True
        keep_session_specified = True

    # Infer cwd from prompt if not specified
    if cwd is None:
        cwd, working = _infer_cwd_from_prompt(working)

    return ExternalAgentConfig(
        enabled=True,
        harness=harness,
        keep_session=keep_session,
        cwd=cwd,
        existing_session_id=session_id,
        prompt=_normalize_prompt(working),
        keep_session_specified=keep_session_specified,
    )


def merge_external_agent_configs(
    *configs: ExternalAgentConfig | None,
) -> ExternalAgentConfig | None:
    """Merge multiple ExternalAgentConfig instances.

    Later configs override earlier ones for explicitly set values.

    Args:
        *configs: Config instances to merge (None values are skipped).

    Returns:
        Merged config or None if all inputs are None.
    """
    merged: ExternalAgentConfig | None = None
    for config in configs:
        if config is None:
            continue
        if merged is None:
            merged = ExternalAgentConfig(
                enabled=config.enabled,
                harness=config.harness,
                keep_session=config.keep_session,
                cwd=config.cwd,
                existing_session_id=config.existing_session_id,
                prompt=config.prompt,
                keep_session_specified=config.keep_session_specified,
            )
            continue

        if config.enabled:
            merged.enabled = True
        if config.harness:
            merged.harness = config.harness
        if config.keep_session_specified:
            merged.keep_session = config.keep_session
            merged.keep_session_specified = True
        if config.cwd:
            merged.cwd = config.cwd
        if config.existing_session_id:
            merged.existing_session_id = config.existing_session_id
            merged.keep_session = True
            merged.keep_session_specified = True
        if config.prompt:
            merged.prompt = config.prompt
    return merged


def parse_external_agent_config(request: Any) -> ExternalAgentConfig | None:
    """Extract external agent config from request extras.

    Looks for config in:
    - request.external_agent
    - request.biz_params.external_agent
    - request.model_extra.external_agent

    Args:
        request: The request object to extract config from.

    Returns:
        ExternalAgentConfig if found and valid, None otherwise.
    """
    raw_config = getattr(request, "external_agent", None)

    if raw_config is None:
        biz_params = getattr(request, "biz_params", None)
        if isinstance(biz_params, dict):
            raw_config = biz_params.get("external_agent")

    model_extra = getattr(request, "model_extra", None)
    if raw_config is None and isinstance(model_extra, dict):
        raw_config = model_extra.get("external_agent")
        if raw_config is None:
            biz_params = model_extra.get("biz_params")
            if isinstance(biz_params, dict):
                raw_config = biz_params.get("external_agent")

    if not isinstance(raw_config, dict):
        return None

    enabled = bool(raw_config.get("enabled"))
    harness = normalize_harness_name(raw_config.get("harness"))
    existing_session_id = (
        raw_config.get("existing_session_id")
        or raw_config.get("session_id")
        or raw_config.get("acp_session_id")
    )
    keep_session = bool(raw_config.get("keep_session")) or bool(
        existing_session_id,
    )
    keep_session_specified = "keep_session" in raw_config or bool(
        existing_session_id,
    )
    cwd = (
        raw_config.get("cwd")
        or raw_config.get("workdir")
        or raw_config.get("working_dir")
    )

    if not enabled or not harness:
        return None

    return ExternalAgentConfig(
        enabled=enabled,
        harness=harness,
        keep_session=keep_session,
        cwd=_strip_quotes(str(cwd)) if cwd else None,
        existing_session_id=_strip_quotes(str(existing_session_id))
        if existing_session_id
        else None,
        keep_session_specified=keep_session_specified,
    )