# -*- coding: utf-8 -*-
"""ACP permission handling built on top of the existing approval service.

This module integrates ACP permission requests with CoPaw's approval system,
enabling user confirmation for dangerous operations performed by external agents.

The permission flow:
1. External agent requests permission via ACP protocol
2. ACPPermissionAdapter creates a pending approval
3. User approves/denies via /approve or /deny commands
4. Result is sent back to the external agent
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from ...security.tool_guard.models import (
    GuardFinding,
    GuardSeverity,
    GuardThreatCategory,
    ToolGuardResult,
)

from .errors import ACPPermissionSuspendedError
from .tool_guard_adapter import get_acp_tool_guard_adapter

logger = logging.getLogger(__name__)

# Tool kinds that are considered read-only and can be auto-approved
READ_ONLY_TOOL_KINDS = frozenset({
    "read",
    "search",
    "list",
    "info",
    "get",
    "fetch",
    "view",
    "grep",
    "glob",
    "find",
})

# Option hints for approval/rejection
ALLOW_OPTION_HINTS = ("allow", "approve", "accept")
REJECT_OPTION_HINTS = ("reject", "deny", "cancel")


@dataclass
class ACPPermissionDecision:
    """Resolved permission result to be returned to the harness.

    Attributes:
        approved: Whether the permission was approved.
        result: The result payload to send back to the harness.
        summary: Human-readable summary of the decision.
    """

    approved: bool
    result: dict[str, Any]
    summary: dict[str, Any] | str = ""


def is_read_only_tool(tool_name: str | None, tool_kind: str | None) -> bool:
    """Check if a tool is considered read-only.

    Read-only tools can be auto-approved if they don't access sensitive paths.

    Args:
        tool_name: The tool name.
        tool_kind: The tool kind.

    Returns:
        True if the tool is read-only.
    """
    kind = str(tool_kind or "").lower()
    name = str(tool_name or "").lower()

    # Check kind first
    if kind in READ_ONLY_TOOL_KINDS:
        return True
    if kind in {"write", "edit", "create", "update", "execute", "run", "shell", "bash", "delete", "remove"}:
        return False

    # Infer from name
    if any(ro in name for ro in ("read", "view", "get", "fetch", "search", "find", "grep", "glob", "list", "ls", "info")):
        return True
    if any(danger in name for danger in ("write", "create", "edit", "update", "execute", "run", "shell", "bash", "delete", "remove", "kill")):
        return False

    # Default to requiring approval for unknown tools
    return False


class ACPPermissionAdapter:
    """Translate ACP permission requests into CoPaw approval decisions.

    This adapter bridges the ACP protocol's permission system with CoPaw's
    existing approval service, enabling consistent user confirmation flows.

    Attributes:
        cwd: Working directory for path resolution.
        require_approval: Whether to require approval for dangerous operations.
    """

    def __init__(self, cwd: str, *, require_approval: bool = False):
        """Initialize the permission adapter.

        Args:
            cwd: Working directory for path resolution.
            require_approval: Whether to require approval for dangerous ops.
        """
        self.cwd = cwd
        self.require_approval = require_approval

    async def resolve_permission(
        self,
        *,
        session_id: str,
        user_id: str,
        channel: str,
        harness: str,
        request_payload: dict[str, Any],
    ) -> ACPPermissionDecision:
        """Resolve one ACP permission request.

        This method:
        1. Extracts tool call info from the request
        2. Checks if auto-approval is possible
        3. Creates a pending approval if needed
        4. Waits for user decision
        5. Returns the result

        Args:
            session_id: The chat session ID.
            user_id: The user ID.
            channel: The channel name.
            harness: The harness name.
            request_payload: The permission request payload.

        Returns:
            The permission decision.
        """
        tool_call = self._extract_tool_call(request_payload)
        tool_name = str(
            tool_call.get("name")
            or request_payload.get("title")
            or "external-agent",
        )
        tool_kind = str(
            tool_call.get("kind") or request_payload.get("kind") or "",
        ).lower()
        options = (
            request_payload.get("options") or tool_call.get("options") or []
        )
        logger.debug(
            "ACP permission request options from harness: %s",
            json.dumps(options, ensure_ascii=False),
        )
        summary = self._build_summary(
            tool_call=tool_call,
            tool_name=tool_name,
            tool_kind=tool_kind,
        )

        allow_option = self._pick_option(
            options,
            ALLOW_OPTION_HINTS,
            fallback_to_first=True,
        )
        reject_option = self._pick_option(
            options,
            REJECT_OPTION_HINTS,
            fallback_to_first=False,
        )
        logger.debug(
            "ACP picked allow_option=%s, reject_option=%s",
            json.dumps(allow_option, ensure_ascii=False) if allow_option else None,
            json.dumps(reject_option, ensure_ascii=False) if reject_option else None,
        )

        # Auto-approve read-only tools
        if self._should_auto_approve(tool_kind, tool_call):
            logger.info(
                "Auto-approving ACP permission: %s (%s)",
                tool_name,
                tool_kind,
            )
            return ACPPermissionDecision(
                approved=True,
                result=self._selected_result(allow_option),
                summary=summary,
            )

        # Run Tool Guard check for external agent tools
        tool_input = tool_call.get("input") or tool_call.get("arguments") or {}
        guard_adapter = get_acp_tool_guard_adapter()
        guard_decision = guard_adapter.check_tool_call(
            harness=harness,
            tool_name=tool_name,
            tool_kind=tool_kind,
            tool_input=tool_input,
            cwd=self.cwd,
        )

        # Block if Tool Guard found HIGH/CRITICAL issues
        if not guard_decision.allowed:
            logger.warning(
                "ACP tool guard blocked: harness=%s tool=%s reason=%s",
                harness,
                tool_name,
                guard_decision.block_reason,
            )
            return ACPPermissionDecision(
                approved=False,
                result=self._selected_result(reject_option),
                summary={
                    "type": "acp_guard_blocked",
                    "harness": harness,
                    "tool_name": tool_name,
                    "reason": guard_decision.block_reason,
                    "guard_result": guard_decision.guard_result.to_dict() if guard_decision.guard_result else None,
                },
            )

        # Skip approval if not required and no guard warnings
        if not self.require_approval and not guard_decision.requires_approval:
            logger.info(
                "Allowing ACP permission without approval: %s (%s)",
                tool_name,
                tool_kind,
            )
            return ACPPermissionDecision(
                approved=True,
                result=self._selected_result(allow_option),
                summary=summary,
            )

        # Suspend execution: raise ACPPermissionSuspendedError so the runtime
        # can exit the event loop cleanly, save the session, and present the
        # choice to the user as a chat message. The caller resumes after the
        # user decides.
        target_text = str(
            tool_call.get("path")
            or tool_call.get("command")
            or tool_call.get("description")
            or ""
        )[:120] or None
        logger.info(
            "ACP suspending permission for user decision: tool=%s kind=%s harness=%s",
            tool_name,
            tool_kind,
            harness,
        )
        raise ACPPermissionSuspendedError(
            payload=request_payload,
            options=options,
            tool_name=tool_name,
            tool_kind=tool_kind,
            target=target_text,
            harness=harness,
        )

    def _extract_tool_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Extract tool call info from permission request payload."""
        if isinstance(payload.get("toolCall"), dict):
            return payload["toolCall"]
        if isinstance(payload.get("tool_call"), dict):
            return payload["tool_call"]
        if isinstance(payload.get("content"), dict):
            return payload["content"]
        return payload

    def _should_auto_approve(
        self,
        tool_kind: str,
        tool_call: dict[str, Any],
    ) -> bool:
        """Check if this tool call should be auto-approved."""
        if not is_read_only_tool(tool_call.get("name"), tool_kind):
            return False

        # For read-only tools, we could check path sensitivity here
        # For now, we auto-approve all read-only tools
        return True

    def _pick_option(
        self,
        options: list[dict[str, Any]],
        hints: tuple[str, ...],
        *,
        fallback_to_first: bool,
    ) -> dict[str, Any] | None:
        """Pick an option from the list based on hints."""
        for option in options:
            if not isinstance(option, dict):
                continue
            values = " ".join(
                str(option.get(key) or "") for key in ("optionId", "kind", "title", "id")
            ).lower()
            if any(hint in values for hint in hints):
                return option
        return options[0] if fallback_to_first and options else None

    def resolve_option_by_id(
        self,
        options: list[dict[str, Any]],
        option_id: str,
    ) -> dict[str, Any] | None:
        """Resolve a user-chosen option by optionId or shorthand keyword.

        Tries in order:
        1. Exact ``optionId`` match (case-insensitive).
        2. Shorthand: ``"allow"`` / ``"approve"`` / ``"yes"`` → first allow option.
        3. Shorthand: ``"deny"`` / ``"reject"`` / ``"no"`` → first reject option.
        4. Partial substring match against optionId / title.

        Args:
            options: The list of option dicts from the harness.
            option_id: The value the user (or model) provided.

        Returns:
            The matching option dict, or ``None`` if nothing matches.
        """
        key = option_id.strip().lower()

        # 1. Exact match
        for opt in options:
            if not isinstance(opt, dict):
                continue
            if str(opt.get("optionId") or opt.get("id") or "").lower() == key:
                return opt

        # 2. Shorthand allow
        if key in ("allow", "approve", "yes", "ok", "proceed"):
            return self._pick_option(options, ALLOW_OPTION_HINTS, fallback_to_first=True)

        # 3. Shorthand deny
        if key in ("deny", "reject", "no", "cancel", "block"):
            return self._pick_option(options, REJECT_OPTION_HINTS, fallback_to_first=False)

        # 4. Partial substring match against optionId or title
        for opt in options:
            if not isinstance(opt, dict):
                continue
            haystack = " ".join(
                str(opt.get(f) or "") for f in ("optionId", "title", "name", "id")
            ).lower()
            if key in haystack:
                return opt

        return None

    def _selected_result(
        self,
        option: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the result payload for the selected option."""
        if option is None:
            logger.debug("ACP _selected_result called with None option, returning cancelled")
            return {"outcome": {"outcome": "cancelled"}}

        option_id = option.get("optionId") or option.get("id") or option.get("kind") or "selected"
        logger.debug(
            "ACP _selected_result option=%s, extracted optionId=%s",
            json.dumps(option, ensure_ascii=False),
            option_id,
        )
        return {
            "outcome": {
                "outcome": "selected",
                "optionId": option_id,
            },
        }

    def _build_summary(
        self,
        *,
        tool_call: dict[str, Any],
        tool_name: str,
        tool_kind: str,
    ) -> dict[str, Any]:
        """Build structured summary for frontend i18n rendering."""
        target = (
            tool_call.get("path")
            or tool_call.get("target")
            or tool_call.get("command")
            or tool_call.get("description")
            or tool_call.get("input")
            or ""
        )
        target_text = str(target).strip()
        if len(target_text) > 240:
            target_text = target_text[:240] + "..."

        harness = tool_call.get("harness") or "external-agent"

        return {
            "type": "acp_approval_summary",
            "harness": str(harness),
            "tool_name": tool_name,
            "tool_kind": tool_kind or "unknown",
            "target": target_text or None,
        }

    def _build_tool_guard_result(
        self,
        *,
        tool_name: str,
        tool_kind: str,
        summary: dict[str, Any] | str,
        tool_call: dict[str, Any],
    ) -> ToolGuardResult:
        """Build a ToolGuardResult for the approval service."""
        description = (
            summary
            if isinstance(summary, str)
            else str(summary.get("tool_name") or summary)
        )
        severity = self._severity_for_kind(tool_kind)
        finding = GuardFinding(
            id="acp_permission",
            rule_id="acp_permission_request",
            category=GuardThreatCategory.CODE_EXECUTION,
            severity=severity,
            title="ACP Permission Request",
            description=description,
            tool_name=tool_name,
            param_name="tool_call",
            matched_value=str(tool_call)[:200],
            guardian="acp_permission_adapter",
        )
        return ToolGuardResult(
            tool_name=tool_name,
            params={"tool_kind": tool_kind, "tool_call": tool_call},
            findings=[finding],
            guardians_used=["acp_permission_adapter"],
        )

    def _severity_for_kind(self, tool_kind: str) -> GuardSeverity:
        """Map tool_kind to an appropriate severity level."""
        kind = str(tool_kind or "").lower()
        if kind in {"execute", "run", "shell", "bash", "delete", "remove"}:
            return GuardSeverity.HIGH
        if kind in {"write", "edit", "create", "update"}:
            return GuardSeverity.MEDIUM
        # read-only or unknown kinds
        return GuardSeverity.LOW


def build_prompt_approval_artifacts(
    *,
    harness: str,
    prompt_text: str,
    cwd: str,
) -> tuple[dict[str, Any], ToolGuardResult, str]:
    """Build approval metadata for a host-side ACP prompt preapproval.

    This is used when a harness cannot be trusted to request permission
    callbacks before performing dangerous actions.

    Args:
        harness: The harness name.
        prompt_text: The prompt text.
        cwd: Working directory.

    Returns:
        Tuple of (summary, result, waiting_text).
    """
    adapter = ACPPermissionAdapter(cwd=cwd, require_approval=True)
    tool_name = f"ACP/{harness}"
    tool_kind = "external_prompt"
    tool_call = {
        "name": tool_name,
        "kind": tool_kind,
        "harness": harness,
        "description": prompt_text,
        "target": cwd,
        "input": {
            "prompt": prompt_text,
            "cwd": cwd,
        },
    }
    summary = adapter._build_summary(  # pylint: disable=protected-access
        tool_call=tool_call,
        tool_name=tool_name,
        tool_kind=tool_kind,
    )
    result = (
        adapter._build_tool_guard_result(  # pylint: disable=protected-access
            tool_name=tool_name,
            tool_kind=tool_kind,
            summary=summary,
            tool_call=tool_call,
        )
    )
    waiting_text = (
        f"⏳ Waiting for approval / 等待审批\n\n"
        f"Harness: {harness}\n"
        f"Request: {prompt_text[:200]}{'...' if len(prompt_text) > 200 else ''}\n"
        f"CWD: {cwd}\n\n"
        f"Type `/approve` to approve, or send any other message to deny.\n"
        f"输入 `/approve` 批准执行，或发送任意消息拒绝。"
    )
    return summary, result, waiting_text