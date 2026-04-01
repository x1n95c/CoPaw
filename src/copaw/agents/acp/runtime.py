# -*- coding: utf-8 -*-
"""ACP runtime built on a bidirectional stdio transport.

This module provides the core ACP protocol implementation, managing
communication with external agent harnesses via JSON-RPC over stdio.

Key responsibilities:
- Process lifecycle management
- Session creation and loading
- Prompt execution with streaming events
- Permission request handling
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
from typing import Any, Awaitable, Callable, Optional

from .config import ACPHarnessConfig
from .errors import ACPErrors, ACPProtocolError, ACPTransportError, ACPPermissionSuspendedError
from .permissions import ACPPermissionAdapter, is_read_only_tool
from .transport import (
    ACPTransport,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
)
from .types import AcpEvent, SuspendedPermission

logger = logging.getLogger(__name__)

# Type aliases for handlers
PermissionHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
EventHandler = Callable[[AcpEvent], Awaitable[None]]

# Default allow/reject options used for host-side permission prompts
_DEFAULT_PERMISSION_OPTIONS = [
    {"optionId": "allow", "kind": "allow", "title": "Allow"},
    {"optionId": "reject", "kind": "reject", "title": "Reject"},
]


class ACPRuntime:
    """Manage one ACP harness process and one active chat session.

    This class handles the complete lifecycle of ACP communication:
    1. Starting the harness process
    2. Performing the initialize handshake
    3. Creating/loading sessions
    4. Sending prompts and handling events
    5. Processing permission requests
    6. Closing the harness

    Attributes:
        harness_name: Name of the harness.
        harness_config: Harness configuration.
        transport: Low-level transport for communication.
        capabilities: Harness capabilities from initialize handshake.
    """

    PROTOCOL_VERSION = 1
    PROMPT_TIMEOUT_SECONDS = 1800.0  # 30 minutes
    PROMPT_DRAIN_GRACE_SECONDS = 1.0
    CANCEL_GRACE_SECONDS = 0.5

    def __init__(self, harness_name: str, harness_config: ACPHarnessConfig):
        """Initialize the runtime.

        Args:
            harness_name: Name of the harness.
            harness_config: Harness configuration.
        """
        self.harness_name = harness_name
        self.harness_config = harness_config
        self.transport = ACPTransport(harness_name, harness_config)
        self.capabilities: dict[str, Any] = {}
        self._cwd: str = ""
        self._require_approval = False
        self._preapproved = False
        self._permission_request_seen = False
        self._permission_broker_verified = False
        self._unsafe_tool_violation_message: str | None = None
        # Suspend/resume state for non-blocking permission flow
        self._suspended_permission: SuspendedPermission | None = None
        self._prompt_task: asyncio.Task | None = None

    # -----------------------------------------------------------------------
    # Lifecycle Methods
    # -----------------------------------------------------------------------

    async def start(self, cwd: str) -> None:
        """Start the harness and perform initialize handshake.

        Args:
            cwd: Working directory for the harness.

        Raises:
            ACPTransportError: If the harness fails to start.
        """
        self._cwd = cwd
        await self.transport.start(cwd=Path(cwd))
        response = await self.transport.send_request(
            "initialize",
            {
                "protocolVersion": self.PROTOCOL_VERSION,
                "clientCapabilities": {
                    "fs": {"readTextFile": True, "writeTextFile": True},
                    "terminal": True,
                    "requestPermission": True,
                },
                "clientInfo": {
                    "name": "CoPaw",
                    "version": "1.0.0",
                },
            },
            timeout=30.0,
        )
        if response.is_error:
            raise ACPTransportError(
                f"initialize failed for {self.harness_name}: {response.error}",
                harness=self.harness_name,
            )

        result = response.result or {}
        agent_capabilities = (
            result.get("agentCapabilities") or result.get("capabilities") or {}
        )
        if isinstance(agent_capabilities, dict):
            self.capabilities = agent_capabilities

    async def close(self) -> None:
        """Shutdown the harness transport."""
        await self.transport.close()

    # -----------------------------------------------------------------------
    # Session Methods
    # -----------------------------------------------------------------------

    async def new_session(self, cwd: str) -> str:
        """Create a new ACP session.

        Args:
            cwd: Working directory for the session.

        Returns:
            The new session ID.

        Raises:
            ACPTransportError: If session creation fails.
        """
        response = await self.transport.send_request(
            "session/new",
            {
                "cwd": cwd,
                "mcpServers": [],
            },
            timeout=60.0,
        )
        if response.is_error:
            raise ACPTransportError(
                "session/new failed for "
                f"{self.harness_name}: {response.error}",
                harness=self.harness_name,
            )
        session_id = (response.result or {}).get("sessionId")
        if not session_id:
            raise ACPProtocolError(
                "session/new response did not include sessionId",
                harness=self.harness_name,
            )
        return str(session_id)

    async def load_session(self, session_id: str, cwd: str) -> str:
        """Load an existing ACP session.

        Args:
            session_id: The session ID to load.
            cwd: Working directory for the session.

        Returns:
            The loaded session ID.

        Raises:
            ACPTransportError: If session loading fails.
        """
        response = await self.transport.send_request(
            "session/load",
            {
                "sessionId": session_id,
                "cwd": cwd,
                "mcpServers": [],
            },
            timeout=60.0,
        )
        if response.is_error:
            raise ACPTransportError(
                "session/load failed for "
                f"{self.harness_name}: {response.error}",
                harness=self.harness_name,
            )
        loaded_id = (response.result or {}).get("sessionId") or session_id
        return str(loaded_id)

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List available sessions from the harness.

        Returns:
            List of session info dictionaries.

        Note:
            Not all harnesses support this method.
        """
        try:
            response = await self.transport.send_request(
                "session/list",
                {},
                timeout=30.0,
            )
            if response.is_error:
                logger.warning(
                    "session/list failed for %s: %s",
                    self.harness_name,
                    response.error,
                )
                return []
            return response.result or []
        except ACPTransportError as e:
            logger.warning("session/list not supported: %s", e)
            return []

    # -----------------------------------------------------------------------
    # Prompt Execution
    # -----------------------------------------------------------------------

    async def prompt(  # pylint: disable=too-many-branches,too-many-statements
        self,
        *,
        chat_id: str,
        session_id: str,
        prompt_blocks: list[dict[str, Any]],
        permission_handler: PermissionHandler,
        on_event: EventHandler,
        timeout: float = PROMPT_TIMEOUT_SECONDS,
        require_approval: bool = False,
        preapproved: bool = False,
        permission_broker_verified: bool = False,
    ) -> None:
        """Send a prompt and stream updates until the turn completes or suspends.

        This is the main execution method that:
        1. Sends the prompt to the harness
        2. Processes incoming events (tool calls, messages, etc.)
        3. Handles permission requests — either auto-approves or suspends
        4. Emits events to the handler
        5. Waits for run_finished (unless suspended for user permission)

        When a permission request requires user interaction, this method
        returns early without emitting run_finished. The caller detects this
        via self._suspended_permission and should:
        - Save the session
        - Return ACPRunResult(suspended_permission=...) to spawn_agent
        - Present options to the user
        - Call resume_prompt_after_permission() once user decides

        Args:
            chat_id: The chat session ID.
            session_id: The ACP session ID.
            prompt_blocks: The prompt content blocks.
            permission_handler: Handler for permission requests.
            on_event: Handler for ACP events.
            timeout: Maximum execution time.
            require_approval: Whether to require approval for dangerous ops.
            preapproved: Whether the prompt was pre-approved.
            permission_broker_verified: Whether the harness is trusted.
        """
        logger.info(
            "ACP prompt starting for %s: chat_id=%s session_id=%s",
            self.harness_name,
            chat_id,
            session_id,
        )

        self._prompt_task = asyncio.create_task(
            self.transport.send_request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": prompt_blocks,
                },
                timeout=timeout,
            ),
        )
        self._require_approval = require_approval
        self._preapproved = preapproved
        self._permission_request_seen = False
        self._permission_broker_verified = permission_broker_verified
        self._unsafe_tool_violation_message = None
        self._suspended_permission = None

        suspended = await self._drain_events_until_done(
            chat_id=chat_id,
            session_id=session_id,
            permission_handler=permission_handler,
            on_event=on_event,
        )

        if not suspended:
            await self._finalize_prompt(chat_id, session_id, on_event)
            self._reset_prompt_state()

    async def resume_prompt_after_permission(
        self,
        *,
        chat_id: str,
        session_id: str,
        permission_result: dict[str, Any],
        on_event: EventHandler,
        permission_handler: PermissionHandler,
    ) -> None:
        """Resume prompt execution after user provides a permission decision.

        This sends the permission result to the harness and re-enters the
        event loop to continue processing until run_finished (or another
        permission suspension).

        Args:
            chat_id: The chat session ID.
            session_id: The ACP session ID.
            permission_result: The result dict to send to the harness
                (e.g. {"outcome": {"outcome": "selected", "optionId": "allow"}}).
            on_event: Handler for ACP events.
            permission_handler: Handler for any subsequent permission requests.

        Raises:
            ACPErrors: If there is no suspended permission to resume.
        """
        if self._suspended_permission is None:
            raise ACPErrors(
                "No suspended permission to resume",
                harness=self.harness_name,
            )

        request_id = self._suspended_permission.request_id
        self._suspended_permission = None
        # Mark that we've seen a permission request this turn to prevent
        # subsequent auto-approval from triggering another permission request
        self._permission_request_seen = True

        logger.info(
            "ACP resuming permission for %s: request_id=%s result=%s",
            self.harness_name,
            request_id,
            permission_result,
        )

        # Send the user's decision back to the harness
        await self.transport.send_result(request_id, permission_result)

        await on_event(
            AcpEvent(
                type="permission_resolved",
                chat_id=chat_id,
                session_id=session_id,
                payload={
                    "summary": (
                        "权限已处理，继续执行 / "
                        "Permission resolved, resuming execution."
                    ),
                },
            ),
        )

        # Continue the event loop
        suspended = await self._drain_events_until_done(
            chat_id=chat_id,
            session_id=session_id,
            permission_handler=permission_handler,
            on_event=on_event,
        )

        if not suspended:
            await self._finalize_prompt(chat_id, session_id, on_event)
            self._reset_prompt_state()

    async def _drain_events_until_done(  # pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks
        self,
        *,
        chat_id: str,
        session_id: str,
        permission_handler: PermissionHandler,
        on_event: EventHandler,
    ) -> bool:
        """Run the event loop until run_finished, task done, or permission suspension.

        Returns:
            True if suspended waiting for user permission, False if completed.
        """
        assert self._prompt_task is not None, "prompt_task must be set before draining"
        prompt_task = self._prompt_task
        run_finished_received = False
        loop = asyncio.get_running_loop()
        cancel_deadline: float | None = None

        while True:
            try:
                incoming = await asyncio.wait_for(
                    self.transport.incoming.get(),
                    timeout=(
                        self.PROMPT_DRAIN_GRACE_SECONDS
                        if prompt_task.done()
                        else 0.1
                    ),
                )
            except asyncio.TimeoutError:
                if (
                    cancel_deadline is not None
                    and not prompt_task.done()
                    and loop.time() >= cancel_deadline
                ):
                    if self.transport.is_running():
                        await self.transport.terminate_with_error(
                            self._unsafe_tool_violation_message
                            or (
                                f"ACP harness {self.harness_name} "
                                "ignored cancellation"
                            ),
                        )
                    cancel_deadline = None
                if prompt_task.done():
                    logger.debug(
                        "ACP prompt task done for %s, draining remaining "
                        "notifications",
                        self.harness_name,
                    )
                    # Drain remaining notifications
                    drain_count = 0
                    while True:
                        try:
                            incoming = await asyncio.wait_for(
                                self.transport.incoming.get(),
                                timeout=0.5,
                            )
                            drain_count += 1
                            if isinstance(incoming, JSONRPCNotification):
                                update = (
                                    incoming.params.get("update")
                                    or incoming.params
                                )
                                if isinstance(update, dict):
                                    update_type = (
                                        update.get("sessionUpdate")
                                        or update.get("type")
                                        or update.get("updateType")
                                        or ""
                                    )
                                    if str(update_type).lower() == "run_finished":
                                        run_finished_received = True
                                await self._handle_notification(
                                    chat_id=chat_id,
                                    session_id=session_id,
                                    notification=incoming,
                                    permission_handler=permission_handler,
                                    on_event=on_event,
                                )
                        except asyncio.TimeoutError:
                            break
                    logger.debug(
                        "ACP drained %d notifications for %s",
                        drain_count,
                        self.harness_name,
                    )
                    break
                continue

            if isinstance(incoming, JSONRPCRequest):
                logger.info(
                    "ACP received request from %s: method=%s params=%s",
                    self.harness_name,
                    incoming.method,
                    incoming.params,
                )
                if "permission" in incoming.method.lower():
                    self._permission_request_seen = True
                try:
                    await self._handle_request(
                        chat_id=chat_id,
                        session_id=session_id,
                        request=incoming,
                        permission_handler=permission_handler,
                        on_event=on_event,
                    )
                except ACPPermissionSuspendedError as exc:
                    # Fill in the JSON-RPC request ID so resume can reply
                    if exc.request_id is None:
                        exc.request_id = incoming.id
                    self._suspended_permission = SuspendedPermission(
                        request_id=exc.request_id,
                        payload=exc.payload,
                        options=exc.options,
                        harness=exc.harness or self.harness_name,
                        tool_name=exc.tool_name,
                        tool_kind=exc.tool_kind,
                        target=exc.target,
                    )
                    logger.info(
                        "ACP prompt suspended for %s: request_id=%s tool=%s",
                        self.harness_name,
                        exc.request_id,
                        exc.tool_name,
                    )
                    return True  # suspended
                continue

            # Check for run_finished
            if isinstance(incoming, JSONRPCNotification):
                update = incoming.params.get("update") or incoming.params
                if isinstance(update, dict):
                    update_type = (
                        update.get("sessionUpdate")
                        or update.get("type")
                        or update.get("updateType")
                        or ""
                    )
                    if str(update_type).lower() == "run_finished":
                        run_finished_received = True

            violation = await self._handle_notification(
                chat_id=chat_id,
                session_id=session_id,
                notification=incoming,
                permission_handler=permission_handler,
                on_event=on_event,
            )
            if violation is not None and self._unsafe_tool_violation_message is None:
                self._unsafe_tool_violation_message = violation
                cancel_deadline = loop.time() + self.CANCEL_GRACE_SECONDS
                await on_event(
                    AcpEvent(
                        type="error",
                        chat_id=chat_id,
                        session_id=session_id,
                        payload={"message": violation},
                    ),
                )
                try:
                    await self.transport.send_notification(
                        "session/cancel",
                        {"sessionId": session_id},
                    )
                except ACPTransportError:
                    if self.transport.is_running():
                        await self.transport.terminate_with_error(violation)

        return False  # completed normally

    async def _finalize_prompt(
        self,
        chat_id: str,
        session_id: str,
        on_event: EventHandler,
    ) -> None:
        """Await prompt_task and emit final run_finished event."""
        assert self._prompt_task is not None
        prompt_task = self._prompt_task

        try:
            response = await prompt_task
        except ACPTransportError as exc:
            response = JSONRPCResponse(
                id=None,
                error={"message": str(exc)},
            )
        except asyncio.CancelledError:
            if self._unsafe_tool_violation_message is None:
                raise
            response = JSONRPCResponse(
                id=None,
                error={"message": self._unsafe_tool_violation_message},
            )

        logger.info(
            "ACP prompt completed for %s: error=%s",
            self.harness_name,
            response.is_error,
        )

        if response.is_error and self._unsafe_tool_violation_message is None:
            await on_event(
                AcpEvent(
                    type="error",
                    chat_id=chat_id,
                    session_id=session_id,
                    payload={"message": str(response.error)},
                ),
            )

        # Always emit run_finished
        await on_event(
            AcpEvent(
                type="run_finished",
                chat_id=chat_id,
                session_id=session_id,
                payload={"result": response.result or {}},
            ),
        )

    def _reset_prompt_state(self) -> None:
        """Reset per-turn state after prompt completes normally."""
        self._require_approval = False
        self._preapproved = False
        self._permission_request_seen = False
        self._permission_broker_verified = False
        self._unsafe_tool_violation_message = None
        self._prompt_task = None

    # -----------------------------------------------------------------------
    # Request/Notification Handling
    # -----------------------------------------------------------------------

    async def _handle_request(
        self,
        *,
        chat_id: str,
        session_id: str,
        request: JSONRPCRequest,
        permission_handler: PermissionHandler,
        on_event: EventHandler,
    ) -> None:
        """Handle a harness-initiated request (e.g., permission request, fs operations)."""
        # Normalize method name: replace both '-' and '/' with '_'
        # e.g., 'fs/write_text_file' -> 'fs_write_text_file'
        method = request.method.replace("-", "_").replace("/", "_")
        logger.info(
            "ACP _handle_request from %s: method=%s id=%s",
            self.harness_name,
            request.method,
            request.id,
        )

        # Handle fs operations
        if method.startswith("fs_"):
            try:
                await self._handle_fs_request(
                    request=request,
                    on_event=on_event,
                    chat_id=chat_id,
                    session_id=session_id,
                    permission_handler=permission_handler,
                )
            except ACPPermissionSuspendedError as exc:
                exc.request_id = request.id
                raise
            return

        # Handle terminal operations
        if method.startswith("terminal_"):
            try:
                await self._handle_terminal_request(
                    request=request,
                    on_event=on_event,
                    chat_id=chat_id,
                    session_id=session_id,
                    permission_handler=permission_handler,
                )
            except ACPPermissionSuspendedError as exc:
                exc.request_id = request.id
                raise
            return

        # Handle permission requests (requestPermission from harness)
        if "permission" not in method.lower():
            logger.warning(
                "Unsupported ACP client request from %s: method=%s params=%s",
                self.harness_name,
                request.method,
                request.params,
            )
            await self.transport.send_error(
                request.id,
                code=-32601,
                message=f"Unsupported ACP client request: {request.method}",
            )
            return

        params = request.params or {}
        summary_payload = dict(params)
        summary_payload.setdefault("harness", self.harness_name)
        logger.info(
            "ACP permission request from %s: params=%s",
            self.harness_name,
            params,
        )
        await on_event(
            AcpEvent(
                type="permission_request",
                chat_id=chat_id,
                session_id=session_id,
                payload=summary_payload,
            ),
        )

        # permission_handler may raise ACPPermissionSuspendedError — set
        # request_id so the resume path can reply to the correct request.
        try:
            result = await permission_handler(params)
        except ACPPermissionSuspendedError as exc:
            exc.request_id = request.id
            raise
        logger.info(
            "ACP permission result for %s: result=%s",
            self.harness_name,
            result,
        )
        await self.transport.send_result(request.id, result)

        await on_event(
            AcpEvent(
                type="permission_resolved",
                chat_id=chat_id,
                session_id=session_id,
                payload={
                    "summary": (
                        "外部 Agent 权限请求已处理 / "
                        "External agent permission request resolved."
                    ),
                },
            ),
        )

    async def _handle_fs_request(
        self,
        *,
        request: JSONRPCRequest,
        on_event: EventHandler,
        chat_id: str,
        session_id: str,
        permission_handler: PermissionHandler,
    ) -> None:
        """Handle fs/* requests from harness (read/write file operations)."""
        # Normalize method name: replace both '-' and '/' with '_'
        method = request.method.replace("-", "_").replace("/", "_")
        params = request.params or {}
        logger.info(
            "ACP fs request from %s: method=%s params=%s",
            self.harness_name,
            method,
            params,
        )
        logger.debug(
            "ACP fs request details - path=%s, content_length=%s",
            params.get("path"),
            len(params.get("content", "")) if params.get("content") else 0,
        )

        try:
            if method == "fs_read_text_file":
                result = await self._fs_read_text_file(params)
            elif method == "fs_write_text_file":
                path = params.get("path", "")
                if not await self._request_fs_write_permission(
                    path=path,
                    params=params,
                    request_id=request.id,
                    chat_id=chat_id,
                    session_id=session_id,
                    on_event=on_event,
                    permission_handler=permission_handler,
                ):
                    return
                result = await self._fs_write_text_file(params)
                await on_event(
                    AcpEvent(
                        type="fs_write",
                        chat_id=chat_id,
                        session_id=session_id,
                        payload={
                            "path": path,
                            "summary": f"文件已写入: {path}",
                        },
                    ),
                )
            else:
                logger.warning(
                    "Unknown fs request from %s: method=%s",
                    self.harness_name,
                    method,
                )
                await self.transport.send_error(
                    request.id,
                    code=-32601,
                    message=f"Unknown fs request: {request.method}",
                )
                return

            await self.transport.send_result(request.id, result)
            logger.info(
                "ACP fs result for %s: method=%s success=True",
                self.harness_name,
                method,
            )
        except ACPPermissionSuspendedError:
            raise  # let it propagate to _drain_events_until_done
        except Exception as e:
            logger.error(
                "ACP fs error for %s: method=%s error=%s",
                self.harness_name,
                method,
                e,
            )
            await self.transport.send_error(
                request.id,
                code=-1,
                message=str(e),
            )

    async def _request_fs_write_permission(
        self,
        *,
        path: str,
        params: dict[str, Any],
        request_id: str | int,
        chat_id: str,
        session_id: str,
        on_event: EventHandler,
        permission_handler: PermissionHandler,
    ) -> bool:
        """Ask user to approve a file write. Returns True if approved.

        If the harness already sent a ``requestPermission`` for this turn and
        the user approved it, the write is auto-approved to avoid asking twice
        for the same operation.
        """
        if self._permission_request_seen:
            # Harness already did its own requestPermission handshake this turn
            # and the user approved — don't prompt again for the same write.
            logger.info(
                "ACP fs/writeTextFile auto-approved for %s (requestPermission already seen): path=%s",
                self.harness_name,
                path,
            )
            return True

        content_len = len(params.get("content", "")) if params.get("content") else 0
        permission_payload = {
            "title": "fs/writeTextFile",
            "toolCall": {
                "name": "fs/writeTextFile",
                "kind": "write",
                "harness": self.harness_name,
                "path": path,
                "input": {"path": path, "content_length": content_len},
            },
            "options": _DEFAULT_PERMISSION_OPTIONS,
        }
        await on_event(
            AcpEvent(
                type="permission_request",
                chat_id=chat_id,
                session_id=session_id,
                payload={**permission_payload, "harness": self.harness_name},
            ),
        )
        try:
            decision = await permission_handler(permission_payload)
        except ACPPermissionSuspendedError as exc:
            # Fill in the actual JSON-RPC request ID so the resume path
            # can send the result back to the correct pending request.
            exc.request_id = request_id
            raise
        outcome = decision.get("outcome") or {}
        option_id = outcome.get("optionId", "")
        approved = (
            outcome.get("outcome") == "selected"
            and option_id not in ("reject", "deny", "cancel")
        )
        logger.info(
            "ACP fs/writeTextFile permission for %s: path=%s approved=%s",
            self.harness_name,
            path,
            approved,
        )
        if not approved:
            await self.transport.send_error(
                request_id,
                code=-32600,
                message=f"Permission denied for writing file: {path}",
            )
        return approved

    async def _fs_read_text_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Read a text file and return its content."""
        path = params.get("path")
        if not path:
            raise ValueError("No path provided for fs/read_text_file")

        # Resolve path relative to working directory if needed
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(self._cwd) / file_path
        file_path = file_path.resolve()
        try:
            file_path.relative_to(Path(self._cwd).resolve())
        except ValueError:
            raise ValueError(f"Path traversal denied: {path}")

        logger.info("ACP reading file: %s", file_path)

        try:
            content = file_path.read_text(encoding="utf-8")
            return {"content": content}
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}")

    async def _fs_write_text_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Write content to a text file."""
        path = params.get("path")
        content = params.get("content")
        if not path:
            raise ValueError("No path provided for fs/write_text_file")
        if content is None:
            raise ValueError("No content provided for fs/write_text_file")

        # Resolve path relative to working directory if needed
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(self._cwd) / file_path
        file_path = file_path.resolve()
        try:
            file_path.relative_to(Path(self._cwd).resolve())
        except ValueError:
            raise ValueError(f"Path traversal denied: {path}")

        logger.info("ACP writing file: %s (%d bytes)", file_path, len(content))

        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return {"success": True}
        except Exception as e:
            raise ValueError(f"Failed to write file: {e}")

    async def _handle_terminal_request(
        self,
        *,
        request: JSONRPCRequest,
        on_event: EventHandler,
        chat_id: str,
        session_id: str,
        permission_handler: PermissionHandler,
    ) -> None:
        """Handle terminal/* requests from harness (execute commands)."""
        # Normalize method name: replace both '-' and '/' with '_'
        method = request.method.replace("-", "_").replace("/", "_")
        params = request.params or {}
        logger.info(
            "ACP terminal request from %s: method=%s params=%s",
            self.harness_name,
            method,
            params,
        )

        try:
            if method == "terminal_execute":
                command = params.get("command", "")
                if not await self._request_terminal_permission(
                    command=command,
                    params=params,
                    request_id=request.id,
                    chat_id=chat_id,
                    session_id=session_id,
                    on_event=on_event,
                    permission_handler=permission_handler,
                ):
                    return
                result = await self._terminal_execute(params)
                await on_event(
                    AcpEvent(
                        type="terminal_execute",
                        chat_id=chat_id,
                        session_id=session_id,
                        payload={
                            "command": command,
                            "summary": f"命令已执行: {command}",
                        },
                    ),
                )
            else:
                logger.warning(
                    "Unknown terminal request from %s: method=%s",
                    self.harness_name,
                    method,
                )
                await self.transport.send_error(
                    request.id,
                    code=-32601,
                    message=f"Unknown terminal request: {request.method}",
                )
                return

            await self.transport.send_result(request.id, result)
            logger.info(
                "ACP terminal result for %s: method=%s success=True",
                self.harness_name,
                method,
            )
        except ACPPermissionSuspendedError:
            raise  # let it propagate to _drain_events_until_done
        except Exception as e:
            logger.error(
                "ACP terminal error for %s: method=%s error=%s",
                self.harness_name,
                method,
                e,
            )
            await self.transport.send_error(
                request.id,
                code=-1,
                message=str(e),
            )

    async def _request_terminal_permission(
        self,
        *,
        command: str,
        params: dict[str, Any],
        request_id: str | int,
        chat_id: str,
        session_id: str,
        on_event: EventHandler,
        permission_handler: PermissionHandler,
    ) -> bool:
        """Ask user to approve a terminal command. Returns True if approved.

        If the harness already sent a ``requestPermission`` for this turn and
        the user approved it, the command is auto-approved to avoid asking twice.
        """
        if self._permission_request_seen:
            logger.info(
                "ACP terminal/execute auto-approved for %s (requestPermission already seen): command=%s",
                self.harness_name,
                command,
            )
            return True

        permission_payload = {
            "title": "terminal/execute",
            "toolCall": {
                "name": "terminal/execute",
                "kind": "execute",
                "harness": self.harness_name,
                "command": command,
                "input": {"command": command, "cwd": params.get("cwd", self._cwd)},
            },
            "options": _DEFAULT_PERMISSION_OPTIONS,
        }
        await on_event(
            AcpEvent(
                type="permission_request",
                chat_id=chat_id,
                session_id=session_id,
                payload={**permission_payload, "harness": self.harness_name},
            ),
        )
        try:
            decision = await permission_handler(permission_payload)
        except ACPPermissionSuspendedError as exc:
            exc.request_id = request_id
            raise
        outcome = decision.get("outcome") or {}
        option_id = outcome.get("optionId", "")
        approved = (
            outcome.get("outcome") == "selected"
            and option_id not in ("reject", "deny", "cancel")
        )
        logger.info(
            "ACP terminal/execute permission for %s: command=%s approved=%s",
            self.harness_name,
            command,
            approved,
        )
        if not approved:
            await self.transport.send_error(
                request_id,
                code=-32600,
                message=f"Permission denied for executing command: {command}",
            )
        return approved

    async def _terminal_execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a terminal command."""
        command = params.get("command")
        if not command:
            raise ValueError("No command provided for terminal/execute")

        cwd = params.get("cwd") or self._cwd
        timeout = params.get("timeout", 60)

        logger.info("ACP executing command: %s in %s", command, cwd)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exitCode": proc.returncode or 0,
            }
        except asyncio.TimeoutError:
            proc.kill()
            raise ValueError(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise ValueError(f"Failed to execute command: {e}")

    async def _handle_notification(
        self,
        *,
        chat_id: str,
        session_id: str,
        notification: JSONRPCNotification,
        permission_handler: PermissionHandler,
        on_event: EventHandler,
    ) -> str | None:
        """Handle a harness-initiated notification.

        Returns:
            Violation message if the notification indicates a policy violation.
        """
        logger.debug(
            "ACP notification from %s: method=%s params=%s",
            self.harness_name,
            notification.method,
            notification.params,
        )

        if notification.method not in {"session/update", "sessionUpdate"}:
            return None

        update = notification.params.get("update") or notification.params
        if not isinstance(update, dict):
            return None

        update_type = (
            update.get("sessionUpdate")
            or update.get("type")
            or update.get("updateType")
            or ""
        )
        normalized = str(update_type).lower()
        payload = self._normalize_payload(update)

        if (
            self._unsafe_tool_violation_message is not None
            and normalized not in {"run_finished", "error"}
        ):
            return None

        logger.debug(
            "ACP update from %s: type=%s normalized=%s",
            self.harness_name,
            update_type,
            normalized,
        )

        event_type = {
            "agent_message_chunk": "assistant_chunk",
            "agent_thought_chunk": "thought_chunk",
            "tool_call": "tool_start",
            "tool_call_update": "tool_update",
            "tool_call_end": "tool_end",
            "plan": "plan_update",
            "usage_update": "usage_update",
            "available_commands_update": "commands_update",
            "run_finished": "run_finished",
            "error": "error",
        }.get(normalized)

        if event_type is None:
            logger.debug(
                "Unknown ACP update type from %s: %s",
                self.harness_name,
                normalized,
            )
            return None

        # Check for unapproved dangerous tools
        if event_type in {"tool_start", "tool_update", "tool_end"}:
            if self._should_block_unapproved_tool(payload):
                # For verified harnesses, we trust them to request permission
                # But if they didn't, we need to handle it
                if self._permission_broker_verified:
                    # For verified harnesses, we trust them to request permission
                    # via the standard ACP requestPermission request mechanism.
                    # DO NOT proactively request permission here - it would block
                    # the event loop and prevent processing the incoming requestPermission.
                    # Just log and continue - the harness will send requestPermission
                    # if needed, which will be handled in _handle_request
                    tool_name = payload.get("name") or "unknown"
                    tool_kind = payload.get("kind") or "unknown"
                    logger.debug(
                        "Verified harness %s executing tool %s (kind=%s), "
                        "waiting for requestPermission if needed",
                        self.harness_name,
                        tool_name,
                        tool_kind,
                    )
                else:
                    return self._build_tool_violation_message(payload)

        await on_event(
            AcpEvent(
                type=event_type,  # type: ignore[arg-type]
                chat_id=chat_id,
                session_id=session_id,
                payload=payload,
            ),
        )
        return None

    def _normalize_payload(self, update: dict[str, Any]) -> dict[str, Any]:
        """Normalize update payload to a consistent format."""
        session_update = str(
            update.get("sessionUpdate")
            or update.get("type")
            or update.get("updateType")
            or "",
        ).lower()

        if session_update == "agent_message_chunk":
            content = update.get("content")
            if isinstance(content, dict):
                return {"text": content.get("text") or ""}
            if isinstance(content, list):
                texts = [
                    str(block.get("text") or "")
                    for block in content
                    if isinstance(block, dict) and block.get("text")
                ]
                return {"text": "".join(texts)}
            return {"text": str(content or "")}

        if session_update in {"tool_call", "tool_call_update", "tool_call_end"}:
            tool = update.get("toolCall") or update.get("tool_call") or update
            if not isinstance(tool, dict):
                tool = update
            tool_input = (
                tool.get("input")
                or tool.get("arguments")
                or tool.get("rawInput")
                or {}
            )
            if not isinstance(tool_input, dict):
                tool_input = {"raw": tool_input}
            tool_output = self._extract_tool_output(tool)
            # Try multiple field names for tool name/kind
            tool_name = (
                tool.get("name")
                or tool.get("tool")
                or tool.get("title")
                or tool.get("toolName")
                or tool.get("tool_name")
                or tool.get("function")
                or tool.get("functionName")
                or update.get("title")
                or update.get("name")
                or update.get("toolName")
                or "unknown"
            )
            tool_kind = (
                tool.get("kind")
                or tool.get("type")
                or tool.get("toolKind")
                or tool.get("tool_type")
                or tool.get("category")
                or update.get("kind")
                or update.get("type")
                or "unknown"
            )
            # Log for debugging
            logger.info(
                "ACP tool_call from %s: name=%s kind=%s update=%s",
                self.harness_name,
                tool_name,
                tool_kind,
                update,
            )
            # Infer kind from name if not provided
            if tool_kind == "unknown" and tool_name != "unknown":
                name_lower = str(tool_name).lower()
                if any(ro in name_lower for ro in ("read", "view", "get", "fetch")):
                    tool_kind = "read"
                elif any(ro in name_lower for ro in ("search", "find", "grep", "glob")):
                    tool_kind = "search"
                elif any(ro in name_lower for ro in ("list", "ls")):
                    tool_kind = "list"
                elif any(ro in name_lower for ro in ("write", "create", "edit", "update")):
                    tool_kind = "write"
                elif any(ro in name_lower for ro in ("execute", "run", "shell", "bash")):
                    tool_kind = "execute"
                elif any(ro in name_lower for ro in ("delete", "remove", "rm")):
                    tool_kind = "delete"
            return {
                "id": tool.get("id") or tool.get("toolCallId"),
                "name": tool_name,
                "kind": tool_kind,
                "input": tool_input,
                "output": tool_output,
                "status": tool.get("status"),
                "summary": tool.get("summary"),
                "detail": tool.get("detail"),
            }

        if session_update == "plan":
            plan = update.get("plan") or update.get("content") or update
            return {"plan": plan}

        if session_update == "available_commands_update":
            return {
                "commands": update.get("availableCommands")
                or update.get("commands")
                or [],
            }

        if session_update == "usage_update":
            usage = update.get("usage")
            if isinstance(usage, dict):
                return usage
            return dict(update)

        if session_update == "error":
            return {
                "message": update.get("message")
                or update.get("error")
                or "ACP error",
            }

        return update

    def _extract_tool_output(self, tool: dict[str, Any]) -> Any:
        """Extract tool output from various payload formats."""
        if tool.get("output") is not None:
            return tool.get("output")

        content = tool.get("content")
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                block = item.get("content")
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if text:
                        chunks.append(str(text))
            if chunks:
                return "\n".join(chunks)

        raw_output = tool.get("rawOutput")
        if isinstance(raw_output, dict):
            if raw_output.get("output") is not None:
                return raw_output.get("output")
            if raw_output.get("error") is not None:
                return raw_output.get("error")

        return None

    def _should_block_unapproved_tool(self, payload: dict[str, Any]) -> bool:
        """Check if an unapproved dangerous tool should be blocked."""
        if not self._require_approval:
            return False
        if self._preapproved:
            return False
        if self._permission_request_seen:
            return False
        
        tool_name = payload.get("name") or "unknown"
        tool_kind = payload.get("kind") or "unknown"
        
        # If harness is verified, trust it to request permission before dangerous ops
        # Don't block - wait for the permission request to come through
        if self._permission_broker_verified:
            logger.debug(
                "Verified harness %s executing tool %s (kind=%s), "
                "waiting for permission request if needed",
                self.harness_name,
                tool_name,
                tool_kind,
            )
            return False
        
        # For unverified harnesses, block dangerous tools that didn't request permission
        return not is_read_only_tool(tool_name, tool_kind)

    def _build_tool_violation_message(self, payload: dict[str, Any]) -> str:
        """Build a violation message for an unapproved tool."""
        tool_name = payload.get("name") or "unknown"
        tool_kind = payload.get("kind") or "unknown"
        return (
            f"ACP harness '{self.harness_name}' attempted to execute "
            f"tool '{tool_name}' (kind: {tool_kind}) without permission request. "
            f"This harness is not verified to honor ACP permission broker protocol."
        )