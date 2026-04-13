# -*- coding: utf-8 -*-
"""ACP to ToolResponse adapter helpers for spawn_agent."""

from pathlib import Path
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse


def _text_block(text: str) -> TextBlock:
    return TextBlock(type="text", text=text)


def response_blocks(
    blocks: list[TextBlock],
    *,
    stream: bool = False,
    is_last: bool = True,
) -> ToolResponse:
    return ToolResponse(content=blocks, stream=stream, is_last=is_last)


def response_text(
    text: str,
    *,
    stream: bool = False,
    is_last: bool = True,
) -> ToolResponse:
    return response_blocks([_text_block(text)], stream=stream, is_last=is_last)


def _header_text(*, runner_name: str, execution_cwd: Path) -> str:
    return f"runner: {runner_name} working directory: {execution_cwd}"


def _string(value: Any) -> str:
    return str(value or "").strip()


def _option_parts(option: Any) -> tuple[str, str] | None:
    if not isinstance(option, dict):
        return None
    option_id = _string(option.get("optionId") or option.get("id"))
    title = _string(option.get("title") or option.get("name") or option_id or "option")
    if not title:
        return None
    return title, option_id


def _render_event_text(event: dict[str, Any]) -> str | None:
    event_type = _string(event.get("type")).lower()

    if event_type == "text":
        text = _string(event.get("text"))
        return f"[assistant]\n{text}" if text else None

    if event_type.startswith("tool_"):
        kind = _string(event.get("kind"))
        detail = _string(event.get("detail") or event.get("title"))
        return f"[tool_call] {kind} ({detail})" if kind and detail else None

    if event_type == "status":
        status = _string(event.get("status")) or "unknown"
        if status == "run_finished":
            return None
        summary = _string(event.get("summary"))
        if status == "agent_thinking":
            return summary or "agent thinking..."
        return "\n".join(part for part in [f"[status] {status}", summary] if part)

    if event_type == "permission_request":
        title = _string(event.get("title") or event.get("reason") or "permission request")
        options = [
            f"{name} ({option_id})" if option_id else name
            for parts in (_option_parts(opt) for opt in event.get("options") or [])
            if parts
            for name, option_id in [parts]
        ]
        return "\n".join(
            part
            for part in [
                f"[permission_request] {title}",
                f"options: {', '.join(options)}" if options else "",
            ]
            if part
        )

    if event_type == "error":
        message_text = _string(event.get("message") or "Unknown error")
        return f"[error] {message_text}" if message_text else None

    return None


def _response(text: str | None, *, stream: bool = False, is_last: bool = True) -> ToolResponse | None:
    if not text:
        return None
    return response_text(text, stream=stream, is_last=is_last)


def event_to_stream_response(
    event: dict[str, Any],
    *,
    runner_name: str,
    execution_cwd: Path,
    include_header: bool = False,
) -> ToolResponse | None:
    text = _render_event_text(event)
    if include_header and text:
        text = f"{_header_text(runner_name=runner_name, execution_cwd=execution_cwd)}\n{text}"
    return _response(text, stream=True, is_last=False)


def format_permission_suspended_response(*, suspended_permission: Any) -> ToolResponse:
    details = [
        f"- Agent: `{getattr(suspended_permission, 'agent', 'unknown')}`",
        f"- Tool: `{getattr(suspended_permission, 'tool_name', 'external-agent')}` (kind: `{getattr(suspended_permission, 'tool_kind', 'other')}`)",
    ]
    action = getattr(suspended_permission, "action", None)
    if action:
        details.append(f"- Action: `{action}`")
    paths = list(getattr(suspended_permission, "paths", []) or [])
    if paths:
        details.append("- Files:")
        details.extend(f"  - `{path}`" for path in paths)
    else:
        target = getattr(suspended_permission, "target", None)
        if target:
            details.append(f"- Target: `{target}`")
    command = getattr(suspended_permission, "command", None)
    if command:
        details.append(f"- Command: `{command}`")
    summary = getattr(suspended_permission, "summary", None)
    if summary:
        details.append(f"- Summary: {summary}")

    options = [
        f"  - **{name}** (`{option_id}`)" if option_id else f"  - **{name}**"
        for parts in (_option_parts(opt) for opt in getattr(suspended_permission, "options", []) or [])
        if parts
        for name, option_id in [parts]
    ]

    text = (
        "🔐 **External Agent Permission Request**\n\n"
        "User confirmation is required before the external agent can continue.\n\n"
        + "\n".join(details)
        + ("\n\nOptions:\n" + "\n".join(options) if options else "")
        + "\n\nReply with one exact option id using `spawn_agent(action=\"respond\", runner=..., message=...)`."
    )
    return response_text(text)


def format_run_completion_response(
    *,
    runner_name: str,
    execution_cwd: Path,
    final_event: dict[str, Any] | None,
) -> ToolResponse:
    text = _render_event_text(final_event or {}) if final_event is not None else None
    return response_text(
        f"{_header_text(runner_name=runner_name, execution_cwd=execution_cwd)}\n{text or 'completed without text output'}"
    )


def format_close_response(*, runner_name: str, closed: bool) -> ToolResponse:
    return response_text(
        f"Closed the bound ACP session for runner '{runner_name}'."
        if closed
        else f"No bound ACP session found for runner '{runner_name}' in the current chat."
    )
