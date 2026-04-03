# -*- coding: utf-8 -*-
"""Shared data types for ACP runtime integration.

This module defines the core data structures used throughout the ACP
integration, including event types, sessions, and run results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


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


@dataclass
class ExternalAgentConfig:
    """Structured config for one external ACP agent request."""

    enabled: bool
    harness: str
    keep_session: bool = False
    cwd: str | None = None
    existing_session_id: str | None = None
    prompt: str | None = None
    keep_session_specified: bool = False
    preapproved: bool = False


@dataclass
class AcpEvent:
    """Internal ACP event emitted by runtime and consumed by handlers."""

    type: ACPEventType
    chat_id: str
    session_id: str | None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuspendedPermission:
    """Pending permission request waiting for user decision."""

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
    """Runtime state for one chat-bound ACP conversation."""

    chat_id: str
    harness: str
    acp_session_id: str
    cwd: str
    keep_session: bool
    capabilities: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=utc_now)
    runtime: Any | None = None
    suspended_permission: SuspendedPermission | None = None


@dataclass
class ACPRunResult:
    """Summary returned after one ACP turn completes."""

    harness: str
    session_id: str | None
    keep_session: bool
    cwd: str
    suspended_permission: SuspendedPermission | None = None
