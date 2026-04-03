# -*- coding: utf-8 -*-
"""ACP Tool Guard Adapter - Integrates external agent tool calls with ToolGuard.

This module bridges the ACP protocol's tool execution events with CoPaw's
ToolGuardEngine, enabling security checks for tools executed by external
agents (like qwen, opencode, gemini).

The flow:
1. External agent sends tool_call event via ACP
2. ACPToolGuardAdapter intercepts and runs ToolGuardEngine checks
3. If issues found:
   - HIGH/CRITICAL: Auto-block or require approval
   - MEDIUM/LOW: Log warning, may require approval
4. Result is passed to permission handler for final decision
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from ...security.tool_guard import ToolGuardEngine, get_guard_engine
from ...security.tool_guard.models import GuardSeverity, ToolGuardResult

logger = logging.getLogger(__name__)


@dataclass
class ACPToolGuardDecision:
    """Decision from ACP tool guard check.

    Attributes:
        allowed: Whether the tool call is allowed.
        guard_result: The ToolGuardResult if checking was performed.
        block_reason: Human-readable reason if blocked.
        requires_approval: Whether user approval is needed.
    """

    allowed: bool
    guard_result: Optional[ToolGuardResult] = None
    block_reason: str | None = None
    requires_approval: bool = False


class ACPToolGuardAdapter:
    """Adapter to run ToolGuard checks on ACP external agent tool calls.

    This class wraps the ToolGuardEngine and provides ACP-specific
    handling for tool calls from external agents.

    Usage:
        adapter = ACPToolGuardAdapter()
        decision = adapter.check_tool_call(
            harness="qwen",
            tool_name="execute_shell_command",
            tool_kind="execute",
            tool_input={"command": "rm -rf /"},
        )
        if not decision.allowed:
            # Block the tool call
        elif decision.requires_approval:
            # Request user approval
    """

    # Tool kinds that map to specific guard checks
    DANGEROUS_KINDS = frozenset({
        "write",
        "edit",
        "create",
        "update",
        "execute",
        "run",
        "shell",
        "bash",
        "delete",
        "remove",
    })

    # Tool name prefixes that indicate external agent tools
    ACP_TOOL_PREFIXES = ("acp_", "external_", "harness_")

    def __init__(self, *, guard_engine: Optional[ToolGuardEngine] = None):
        """Initialize the adapter.

        Args:
            guard_engine: Optional custom guard engine. Defaults to global.
        """
        self._engine = guard_engine or get_guard_engine()

    def check_tool_call(
        self,
        *,
        harness: str,
        tool_name: str,
        tool_kind: str,
        tool_input: dict[str, Any],
        cwd: str,
        preapproved: bool = False,
    ) -> ACPToolGuardDecision:
        """Check an ACP tool call against security rules.

        Args:
            harness: The harness name (e.g., "qwen", "opencode").
            tool_name: The tool being called.
            tool_kind: The kind of tool operation.
            tool_input: The tool input parameters.
            cwd: Working directory for path resolution.
            preapproved: Whether the call was pre-approved.

        Returns:
            ACPToolGuardDecision with the check result.
        """
        if not self._engine.enabled:
            return ACPToolGuardDecision(allowed=True)

        if preapproved:
            return ACPToolGuardDecision(allowed=True)

        # Normalize tool name for guard engine
        normalized_tool_name = self._normalize_tool_name(tool_name, harness)

        # Map ACP tool kind to guard parameters
        guard_params = self._build_guard_params(tool_kind, tool_input, cwd)

        # Run guard engine
        guard_result = self._engine.guard(normalized_tool_name, guard_params)

        if guard_result is None:
            return ACPToolGuardDecision(allowed=True)

        # Analyze findings
        high_critical = [
            f for f in guard_result.findings
            if f.severity in (GuardSeverity.CRITICAL, GuardSeverity.HIGH)
        ]

        if high_critical:
            # Auto-block HIGH/CRITICAL findings
            reasons = [f.description for f in high_critical[:3]]
            block_reason = (
                f"ACP tool guard blocked {normalized_tool_name} from {harness}: "
                + "; ".join(reasons)
            )
            logger.warning(
                "ACP tool guard blocked: harness=%s tool=%s findings=%d",
                harness,
                normalized_tool_name,
                len(high_critical),
            )
            return ACPToolGuardDecision(
                allowed=False,
                guard_result=guard_result,
                block_reason=block_reason,
                requires_approval=False,  # Already blocked
            )

        # MEDIUM/LOW findings: require approval but don't auto-block
        if guard_result.findings:
            logger.info(
                "ACP tool guard warnings: harness=%s tool=%s findings=%d",
                harness,
                normalized_tool_name,
                len(guard_result.findings),
            )
            return ACPToolGuardDecision(
                allowed=True,
                guard_result=guard_result,
                requires_approval=True,
            )

        # No findings: safe to proceed
        return ACPToolGuardDecision(
            allowed=True,
            guard_result=guard_result,
            requires_approval=self._should_require_approval(tool_kind, tool_input),
        )

    def _normalize_tool_name(self, tool_name: str, harness: str) -> str:
        """Normalize ACP tool name for guard engine matching.

        Maps external agent tool names to CoPaw's internal tool naming
        convention so existing guard rules can match them.
        """
        # Remove ACP prefixes
        name = str(tool_name or "unknown").lower()
        for prefix in self.ACP_TOOL_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]

        # Map common external tool names to internal equivalents
        tool_mappings = {
            "readfile": "read_file",
            "writefile": "write_file",
            "editfile": "edit_file",
            "execute_shell": "execute_shell_command",
            "shell": "execute_shell_command",
            "bash": "execute_shell_command",
            "run_command": "execute_shell_command",
            "delete_file": "delete_file",
            "remove_file": "delete_file",
        }

        mapped = tool_mappings.get(name)
        if mapped:
            return mapped

        # Prefix with harness name for unknown tools
        if not name.startswith(harness):
            return f"{harness}_{name}"

        return name

    def _build_guard_params(
        self,
        tool_kind: str,
        tool_input: dict[str, Any],
        cwd: str,
    ) -> dict[str, Any]:
        """Build guard engine parameters from ACP tool input.

        Maps ACP tool input format to CoPaw's guard parameter convention.
        """
        params = dict(tool_input or {})

        # Ensure path parameters are resolved relative to cwd
        path_keys = ("path", "file_path", "filepath", "target", "destination")
        for key in path_keys:
            if key in params and isinstance(params[key], str):
                path = params[key]
                if not path.startswith("/") and not path.startswith("~"):
                    params[key] = f"{cwd}/{path}"

        # Map common ACP input keys to guard engine keys
        key_mappings = {
            "cmd": "command",
            "shell_command": "command",
            "script": "command",
            "content": "text",
            "file_content": "text",
        }

        for old_key, new_key in key_mappings.items():
            if old_key in params and new_key not in params:
                params[new_key] = params[old_key]

        return params

    def _should_require_approval(
        self,
        tool_kind: str,
        tool_input: dict[str, Any],
    ) -> bool:
        """Determine if a tool call should require approval based on kind."""
        kind = str(tool_kind or "").lower()

        # Dangerous kinds always require approval
        if kind in self.DANGEROUS_KINDS:
            return True

        # Check input for dangerous patterns
        input_str = str(tool_input).lower()
        dangerous_patterns = [
            "rm -rf",
            "delete",
            "format",
            "drop table",
            "truncate",
            "sudo",
            "chmod",
            "chown",
        ]
        return any(p in input_str for p in dangerous_patterns)

# Global adapter instance
_acp_tool_guard_adapter: ACPToolGuardAdapter | None = None


def get_acp_tool_guard_adapter() -> ACPToolGuardAdapter:
    """Get the global ACP tool guard adapter."""
    global _acp_tool_guard_adapter
    if _acp_tool_guard_adapter is None:
        _acp_tool_guard_adapter = ACPToolGuardAdapter()
    return _acp_tool_guard_adapter


def init_acp_tool_guard_adapter(
    guard_engine: Optional[ToolGuardEngine] = None,
) -> ACPToolGuardAdapter:
    """Initialize the global ACP tool guard adapter."""
    global _acp_tool_guard_adapter
    _acp_tool_guard_adapter = ACPToolGuardAdapter(guard_engine=guard_engine)
    return _acp_tool_guard_adapter