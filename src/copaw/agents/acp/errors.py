# -*- coding: utf-8 -*-
"""ACP error types for structured error handling.

This module defines specific exception types for ACP operations,
enabling callers to distinguish between different failure modes.
"""
from __future__ import annotations

from typing import Optional


class ACPErrors(Exception):
    """Base exception for all ACP-related errors.

    All ACP exceptions inherit from this base class for easy catching.
    """

    def __init__(self, message: str, *, harness: Optional[str] = None):
        super().__init__(message)
        self.harness = harness


class ACPConfigurationError(ACPErrors):
    """Raised when ACP configuration is invalid or missing.

    Common causes:
    - ACP is disabled in config
    - Requested harness is not configured
    - Requested harness is disabled
    - Required configuration is missing
    """

    pass


class ACPTransportError(ACPErrors):
    """Raised when ACP transport communication fails.

    Common causes:
    - Harness process failed to start
    - Harness process crashed
    - Communication timeout
    - Invalid response from harness
    """

    pass


class ACPProtocolError(ACPErrors):
    """Raised when ACP protocol message is invalid.

    Common causes:
    - Invalid JSON-RPC message format
    - Missing required fields
    - Unexpected message type
    """

    pass


class ACPSessionError(ACPErrors):
    """Raised when session operations fail.

    Common causes:
    - Session not found
    - Session expired
    - Session load failed
    """

    pass


class ACPPermissionSuspendedError(ACPErrors):
    """Raised to suspend prompt execution pending user permission decision.

    When user approval is needed for a permission request, this exception
    is raised instead of blocking. The runtime stores the pending state and
    returns control to the caller, which presents options to the user via chat.

    The full flow:
    1. Permission request arrives from harness
    2. ACPPermissionAdapter raises this exception
    3. Runtime catches it, fills in request_id, stores SuspendedPermission
    4. service.run_turn() detects suspension, saves session, returns result
    5. spawn_agent formats a chat message with options for the user
    6. User replies, model calls spawn_agent with permission_decision
    7. service.resume_permission() sends result to harness and continues

    Attributes:
        request_id: JSON-RPC request ID to reply to (set by runtime).
        payload: The raw permission request payload.
        options: Available option dicts for the user to choose from.
        tool_name: The tool requesting permission.
        tool_kind: The kind of operation (write, execute, etc.).
        target: The target of the operation (path, command, etc.).
    """

    def __init__(
        self,
        *,
        payload: "dict",
        options: "list",
        tool_name: str,
        tool_kind: str,
        target: "Optional[str]" = None,
        harness: "Optional[str]" = None,
    ):
        super().__init__(
            "ACP permission suspended waiting for user decision",
            harness=harness,
        )
        self.request_id: object = None  # filled in by runtime._handle_request
        self.payload = payload
        self.options = options
        self.tool_name = tool_name
        self.tool_kind = tool_kind
        self.target = target