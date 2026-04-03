# -*- coding: utf-8 -*-
"""ACP stdio transport with bidirectional JSON-RPC support.

This module handles the low-level communication with ACP harness processes,
including process lifecycle management and message encoding/decoding.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import ACPHarnessConfig
from .errors import ACPProtocolError, ACPTransportError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON-RPC Message Types
# ---------------------------------------------------------------------------

@dataclass
class JSONRPCResponse:
    """JSON-RPC response envelope.

    Attributes:
        id: Request ID this response corresponds to.
        result: Result payload on success.
        error: Error payload on failure.
    """

    id: str | int | None
    result: Any = None
    error: dict[str, Any] | None = None

    @property
    def is_error(self) -> bool:
        """Check if this response represents an error."""
        return self.error is not None


@dataclass
class JSONRPCRequest:
    """JSON-RPC request envelope initiated by the harness.

    Used for permission requests and other harness-initiated calls.

    Attributes:
        id: Request ID for response correlation.
        method: The method being called.
        params: Method parameters.
    """

    id: str | int
    method: str
    params: dict[str, Any]


@dataclass
class JSONRPCNotification:
    """JSON-RPC notification envelope initiated by the harness.

    Used for streaming events (tool calls, message chunks, etc.).

    Attributes:
        method: The notification method (e.g., "session/update").
        params: Notification parameters.
    """

    method: str
    params: dict[str, Any]


# ---------------------------------------------------------------------------
# Transport Implementation
# ---------------------------------------------------------------------------

class ACPTransport:
    """Manage ACP harness process lifecycle and message routing.

    This class handles:
    - Starting and stopping the harness process
    - Sending JSON-RPC requests and receiving responses
    - Receiving harness-initiated requests and notifications
    - Process lifecycle and error handling

    Attributes:
        harness_name: Name of the harness for logging.
        config: Harness configuration.
        incoming: Queue of incoming requests/notifications from harness.
        stderr_tail: Recent stderr output for debugging.
    """

    STDIO_STREAM_LIMIT = 1024 * 1024  # 1MB
    _MAX_STDERR_BUFFER_LINES = 100

    def __init__(self, harness_name: str, harness_config: ACPHarnessConfig):
        """Initialize the transport.

        Args:
            harness_name: Name of the harness (for logging).
            harness_config: Harness configuration.
        """
        self.harness_name = harness_name
        self.config = harness_config
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._request_id = 0
        self._pending: dict[str, asyncio.Future[JSONRPCResponse]] = {}
        self._incoming: asyncio.Queue[
            JSONRPCRequest | JSONRPCNotification
        ] = asyncio.Queue()
        self._stderr_buffer: deque[str] = deque(
            maxlen=self._MAX_STDERR_BUFFER_LINES,
        )

    @property
    def incoming(self) -> asyncio.Queue[JSONRPCRequest | JSONRPCNotification]:
        """Queue of incoming messages from the harness."""
        return self._incoming

    @property
    def stderr_tail(self) -> list[str]:
        """Recent stderr output (last 20 lines)."""
        return list(self._stderr_buffer[-20:])

    def is_running(self) -> bool:
        """Check if the harness process is still alive."""
        return self._process is not None and self._process.returncode is None

    # -----------------------------------------------------------------------
    # Process Lifecycle
    # -----------------------------------------------------------------------

    async def start(self, cwd: str | Path | None = None) -> None:
        """Spawn the harness process and start background readers.

        Args:
            cwd: Working directory for the harness process.

        Raises:
            ACPTransportError: If the process fails to start.
        """
        if self.is_running():
            await self.close()

        working_dir = Path(cwd or Path.cwd()).expanduser().resolve()
        env = os.environ.copy()
        env.update(self.config.env)
        cmd = [self.config.command, *self.config.args]
        logger.info(
            "Spawning ACP harness %s in %s",
            self.harness_name,
            working_dir,
        )

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=self.STDIO_STREAM_LIMIT,
                cwd=str(working_dir),
                env=env,
            )
        except Exception as exc:
            raise ACPTransportError(
                f"Failed to spawn harness {self.harness_name}: {exc}",
                harness=self.harness_name,
            ) from exc

        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stdout_task.add_done_callback(
            lambda task: self._on_reader_task_done("stdout", task),
        )
        self._stderr_task = asyncio.create_task(self._read_stderr())
        self._stderr_task.add_done_callback(
            lambda task: self._on_reader_task_done("stderr", task),
        )

    async def close(self) -> None:
        """Stop reader tasks and terminate the harness process."""
        await self._cancel_reader_tasks()
        self._cancel_pending_requests()
        await self._shutdown_process()

        self._process = None
        self._stdout_task = None
        self._stderr_task = None

    # -----------------------------------------------------------------------
    # Message Sending
    # -----------------------------------------------------------------------

    async def send_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 60.0,
    ) -> JSONRPCResponse:
        """Send a JSON-RPC request and await its response.

        Args:
            method: The method to call.
            params: Method parameters.
            timeout: Maximum time to wait for response.

        Returns:
            The response from the harness.

        Raises:
            ACPTransportError: If the request times out or transport fails.
        """
        request_id = self._next_request_id()
        payload = self._build_request_payload(
            request_id=request_id,
            method=method,
            params=params,
        )
        future: asyncio.Future[
            JSONRPCResponse
        ] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        try:
            await self._write_payload(payload)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise ACPTransportError(
                "Timed out waiting for "
                f"{method} response from {self.harness_name}",
                harness=self.harness_name,
            ) from exc
        finally:
            self._pending.pop(request_id, None)

    async def send_notification(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        """Send a JSON-RPC notification (no response expected).

        Args:
            method: The notification method.
            params: Notification parameters.
        """
        await self._write_payload(
            self._build_notification_payload(method=method, params=params),
        )

    async def send_result(self, request_id: str | int, result: Any) -> None:
        """Reply to a harness-initiated request with a result.

        Args:
            request_id: The ID of the request being answered.
            result: The result to send.
        """
        await self._write_payload(
            self._build_result_payload(request_id=request_id, result=result),
        )

    async def send_error(
        self,
        request_id: str | int,
        *,
        code: int,
        message: str,
    ) -> None:
        """Reply to a harness-initiated request with an error.

        Args:
            request_id: The ID of the request being answered.
            code: JSON-RPC error code.
            message: Error message.
        """
        await self._write_payload(
            self._build_error_payload(
                request_id=request_id,
                code=code,
                message=message,
            ),
        )

    async def terminate_with_error(self, message: str) -> None:
        """Fail pending requests with message and close the harness.

        Args:
            message: Error message to use for pending requests.
        """
        self._fail_pending(message)
        await self.close()

    # -----------------------------------------------------------------------
    # Internal Methods
    # -----------------------------------------------------------------------

    async def _write_payload(self, payload: dict[str, Any]) -> None:
        """Write a JSON-RPC payload to the harness stdin."""
        if self._process is None or self._process.stdin is None:
            raise ACPTransportError(
                "Harness process is not running",
                harness=self.harness_name,
            )

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        if "id" in payload and "result" in payload:
            logger.debug(
                "ACP sending response to %s: %s",
                self.harness_name,
                data.decode("utf-8")[:500],
            )
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    def _build_request_payload(
        self,
        *,
        request_id: str | int,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

    def _build_notification_payload(
        self,
        *,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

    def _build_result_payload(
        self,
        *,
        request_id: str | int,
        result: Any,
    ) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def _build_error_payload(
        self,
        *,
        request_id: str | int,
        code: int,
        message: str,
    ) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }

    async def _cancel_reader_tasks(self) -> None:
        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _cancel_pending_requests(self) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()

    async def _shutdown_process(self) -> None:
        if self._process is None:
            return

        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
                wait_closed = getattr(
                    self._process.stdin,
                    "wait_closed",
                    None,
                )
                if callable(wait_closed):
                    await wait_closed()
            except Exception as exc:
                logger.debug(
                    "Failed closing stdin for ACP harness %s: %s",
                    self.harness_name,
                    exc,
                )
        try:
            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except ProcessLookupError:
            logger.debug(
                "ACP harness %s already exited before terminate",
                self.harness_name,
            )
        except asyncio.TimeoutError:
            try:
                self._process.kill()
                await self._process.wait()
            except ProcessLookupError:
                logger.debug(
                    "ACP harness %s already exited before kill",
                    self.harness_name,
                )

    def _on_reader_task_done(
        self,
        stream_name: str,
        task: asyncio.Task[None],
    ) -> None:
        """Handle reader task completion (error or cancellation)."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return

        message = (
            f"ACP harness {self.harness_name} {stream_name} "
            f"reader failed: {exc}"
        )
        logger.error(message)
        self._fail_pending(message)

    def _fail_pending(self, message: str) -> None:
        """Fail all pending requests with an error."""
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(ACPTransportError(message))
        self._pending.clear()

    def _next_request_id(self) -> str:
        """Generate the next request ID."""
        self._request_id += 1
        return f"req_{self._request_id}"

    async def _read_stdout(self) -> None:
        """Read and process stdout lines from the harness."""
        if self._process is None or self._process.stdout is None:
            return

        while True:
            line = await self._process.stdout.readline()
            if not line:
                break

            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue

            logger.debug(
                "ACP raw stdout from %s: %s",
                self.harness_name,
                text[:300],
            )

            try:
                message = self._decode_message(text)
            except ACPProtocolError:
                logger.debug(
                    "Ignoring non-JSON ACP stdout line: %s",
                    text[:200],
                )
                continue

            if await self._handle_decoded_message(message):
                continue

            await self._incoming.put(message)

    async def _read_stderr(self) -> None:
        """Read and buffer stderr lines from the harness."""
        if self._process is None or self._process.stderr is None:
            return

        while True:
            line = await self._process.stderr.readline()
            if not line:
                break

            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue

            self._stderr_buffer.append(text)
            logger.debug(
                "ACP harness stderr (%s): %s",
                self.harness_name,
                text,
            )

    async def _handle_decoded_message(
        self,
        message: JSONRPCResponse | JSONRPCRequest | JSONRPCNotification,
    ) -> bool:
        if isinstance(message, JSONRPCResponse):
            logger.debug(
                "ACP response received for %s: id=%s result=%s error=%s",
                self.harness_name,
                message.id,
                type(message.result).__name__ if message.result else None,
                message.error,
            )
            pending = self._pending.get(str(message.id))
            if pending is not None and not pending.done():
                pending.set_result(message)
            return True

        if isinstance(message, JSONRPCRequest):
            logger.info(
                "ACP request from harness %s: id=%s method=%s",
                self.harness_name,
                message.id,
                message.method,
            )
            if message.method.startswith("fs/"):
                logger.debug(
                    "ACP fs request from %s: id=%s method=%s params=%s",
                    self.harness_name,
                    message.id,
                    message.method,
                    json.dumps(message.params, ensure_ascii=False)[:500],
                )
        elif isinstance(message, JSONRPCNotification):
            logger.debug(
                "ACP notification from harness %s: method=%s",
                self.harness_name,
                message.method,
            )

        return False

    def _decode_message(
        self,
        raw: str,
    ) -> JSONRPCResponse | JSONRPCRequest | JSONRPCNotification:
        """Decode a JSON-RPC message from raw text.

        Args:
            raw: The raw JSON-RPC message text.

        Returns:
            Parsed message object.

        Raises:
            ACPProtocolError: If the message is invalid.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ACPProtocolError(
                f"Invalid JSON-RPC payload: {raw}",
                harness=self.harness_name,
            ) from exc

        if not isinstance(data, dict):
            raise ACPProtocolError(
                "JSON-RPC payload must be an object",
                harness=self.harness_name,
            )

        # Request from harness (has method, id, no result/error)
        if (
            "method" in data
            and "id" in data
            and "result" not in data
            and "error" not in data
        ):
            return JSONRPCRequest(
                id=data["id"],
                method=str(data["method"]),
                params=data.get("params") or {},
            )

        # Notification from harness (has method, no id)
        if "method" in data:
            return JSONRPCNotification(
                method=str(data["method"]),
                params=data.get("params") or {},
            )

        # Response (has id)
        if "id" in data:
            return JSONRPCResponse(
                id=data.get("id"),
                result=data.get("result"),
                error=data.get("error"),
            )

        raise ACPProtocolError(
            f"Unknown JSON-RPC payload shape: {raw}",
            harness=self.harness_name,
        )