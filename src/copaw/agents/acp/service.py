# -*- coding: utf-8 -*-
"""High-level ACP service used by tools and runners.

This module provides the main entry point for ACP operations, handling:
- Session management (creation, loading, persistence)
- Permission enforcement
- Event streaming to callers
- Error handling and recovery
"""
from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable, Optional

from .config import ACPConfig, ACPHarnessConfig
from .errors import ACPConfigurationError, ACPSessionError
from .permissions import (
    ACPPermissionAdapter,
    ALLOW_OPTION_HINTS,
    REJECT_OPTION_HINTS,
    build_prompt_approval_artifacts,
)
from .runtime import ACPRuntime
from .session_store import ACPSessionStore, get_session_store
from .types import (
    ACPConversationSession,
    ACPRunResult,
    AcpEvent,
    ExternalAgentConfig,
    SuspendedPermission,
)

logger = logging.getLogger(__name__)


class ACPService:
    """Run ACP turns and manage per-chat harness sessions.

    This is the main service class for ACP operations. It coordinates:
    - Session lifecycle (create, load, persist, close)
    - Permission handling
    - Event streaming
    - Error handling

    Attributes:
        config: ACP configuration.
    """

    def __init__(self, *, config: ACPConfig):
        """Initialize the ACP service.

        Args:
            config: ACP configuration.
        """
        self.config = config
        self._store = ACPSessionStore(save_dir=config.save_dir)

    # -----------------------------------------------------------------------
    # Main Operations
    # -----------------------------------------------------------------------

    async def run_turn(
        self,
        *,
        chat_id: str,
        session_id: str,
        user_id: str,
        channel: str,
        harness: str,
        prompt_blocks: list[dict[str, Any]],
        cwd: str,
        keep_session: bool,
        preapproved: bool = False,
        existing_session_id: str | None = None,
        on_message: Callable[[Any, bool], Awaitable[None]],
    ) -> ACPRunResult:
        """Run one ACP turn and stream projected messages back.

        This is the main entry point for executing an ACP turn. It:
        1. Validates configuration
        2. Gets or creates a conversation session
        3. Sets up permission handling
        4. Runs the prompt
        5. Handles events and errors
        6. Persists or closes the session

        Args:
            chat_id: The chat session ID.
            session_id: The CoPaw session ID (for approvals).
            user_id: The user ID.
            channel: The channel name.
            harness: The harness name.
            prompt_blocks: The prompt content blocks.
            cwd: Working directory.
            keep_session: Whether to keep the session alive.
            preapproved: Whether the request was pre-approved.
            existing_session_id: Existing ACP session ID to resume.
            on_message: Callback for streaming messages.

        Returns:
            Summary of the completed turn.
        """
        harness_config = self._get_harness_config(harness)
        self._enforce_unverified_harness_policy(
            harness=harness,
            harness_config=harness_config,
            prompt_blocks=prompt_blocks,
            preapproved=preapproved,
        )

        conversation, ephemeral = await self._get_or_create_conversation(
            chat_id=chat_id,
            harness=harness,
            cwd=cwd,
            keep_session=keep_session,
            existing_session_id=existing_session_id,
            harness_config=harness_config,
        )

        permission_adapter = ACPPermissionAdapter(
            cwd=conversation.cwd,
            require_approval=self.config.require_approval,
        )

        async def _handle_event(event: AcpEvent) -> None:
            """Handle ACP events by converting to messages."""
            projected = self._project_event(event)
            for message, last in projected:
                logger.debug(
                    "ACP service: calling on_message for event %s, "
                    "msg_id=%s, last=%s",
                    event.type,
                    getattr(message, "id", "?"),
                    last,
                )
                await on_message(message, last)

        async def _resolve_permission(payload: dict[str, Any]) -> dict[str, Any]:
            """Resolve permission requests via the approval service."""
            decision = await permission_adapter.resolve_permission(
                session_id=session_id,
                user_id=user_id,
                channel=channel,
                harness=harness,
                request_payload=payload,
            )
            if decision.summary:
                payload = dict(payload)
                payload["summary"] = decision.summary
            return decision.result

        try:
            await conversation.runtime.prompt(
                chat_id=chat_id,
                session_id=conversation.acp_session_id,
                prompt_blocks=prompt_blocks,
                permission_handler=_resolve_permission,
                on_event=_handle_event,
                require_approval=self.config.require_approval,
                preapproved=preapproved,
                permission_broker_verified=(
                    harness_config.permission_broker_verified
                ),
            )
        finally:
            logger.debug("ACP service: finalizing turn")
            suspended = conversation.runtime._suspended_permission is not None

            if suspended:
                # Turn is suspended waiting for user permission — keep the
                # transport alive and always persist the session so it can
                # be resumed in the next call.
                conversation.suspended_permission = (
                    conversation.runtime._suspended_permission
                )
                await self._store.save(conversation)
            elif ephemeral:
                await conversation.runtime.close()
            else:
                await self._store.save(conversation)

        if conversation.runtime._suspended_permission is not None:
            return ACPRunResult(
                harness=harness,
                session_id=conversation.acp_session_id,
                keep_session=True,
                cwd=conversation.cwd,
                suspended_permission=conversation.runtime._suspended_permission,
            )

        return ACPRunResult(
            harness=harness,
            session_id=conversation.acp_session_id,
            keep_session=keep_session,
            cwd=conversation.cwd,
        )

    async def close_chat_session(
        self,
        *,
        chat_id: str,
        harness: str,
        reason: str,
    ) -> None:
        """Close a persisted ACP chat session.

        Args:
            chat_id: The chat session ID.
            harness: The harness name.
            reason: Reason for closing (for logging).
        """
        logger.info(
            "Closing ACP chat session %s/%s: %s",
            chat_id,
            harness,
            reason,
        )
        existing = await self._store.delete(chat_id, harness)
        if existing is not None and existing.runtime is not None:
            await existing.runtime.close()

    # -----------------------------------------------------------------------
    # Session Management
    # -----------------------------------------------------------------------

    async def list_sessions(
        self,
        chat_id: Optional[str] = None,
        harness: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List available ACP sessions.

        Args:
            chat_id: Optional filter by chat ID.
            harness: Optional filter by harness.

        Returns:
            List of session info dictionaries.
        """
        return await self._store.list_sessions(chat_id=chat_id, harness=harness)

    async def get_session(
        self,
        chat_id: str,
        harness: str,
    ) -> ACPConversationSession | None:
        """Get an active session by chat_id and harness.

        Args:
            chat_id: The chat session ID.
            harness: The harness name.

        Returns:
            The session if found, None otherwise.
        """
        return await self._store.get(chat_id, harness)

    async def load_session_by_acp_id(
        self,
        acp_session_id: str,
    ) -> ACPConversationSession | None:
        """Get a session by its ACP session ID.

        Args:
            acp_session_id: The ACP protocol session ID.

        Returns:
            The session if found, None otherwise.
        """
        return await self._store.get_session_by_acp_id(acp_session_id)

    # -----------------------------------------------------------------------
    # Approval Helpers
    # -----------------------------------------------------------------------

    def build_preapproval_artifacts(
        self,
        *,
        harness: str,
        prompt_text: str,
        cwd: str,
    ) -> tuple[dict[str, Any], Any, str]:
        """Build approval artifacts for pre-approval flow.

        Args:
            harness: The harness name.
            prompt_text: The prompt text.
            cwd: Working directory.

        Returns:
            Tuple of (summary, result, waiting_text).
        """
        return build_prompt_approval_artifacts(
            harness=harness,
            prompt_text=prompt_text,
            cwd=cwd,
        )

    async def resume_permission(
        self,
        *,
        acp_session_id: str,
        option_id: str,
        on_message: Callable[[Any, bool], Awaitable[None]],
    ) -> ACPRunResult:
        """Resume a suspended ACP turn after user provides a permission decision.

        This looks up the session, resolves the chosen option from the
        harness-provided option list, sends the result to the harness, and
        continues running the event loop until the turn completes (or
        suspends again for another permission request).

        Args:
            acp_session_id: The ACP session ID of the suspended session.
            option_id: The chosen option. Can be an exact ``optionId`` from
                the harness (e.g. ``"proceed_once"``, ``"proceed_always"``,
                ``"cancel"``), or a shorthand keyword:
                ``"allow"`` / ``"approve"`` / ``"yes"`` → pick first allow option;
                ``"deny"`` / ``"reject"`` / ``"no"`` → pick first reject option.
            on_message: Callback for streaming messages.

        Returns:
            Summary of the resumed turn.

        Raises:
            ACPSessionError: If the session is not found or has no pending
                permission, or if ``option_id`` does not match any option.
        """
        conversation = await self._store.get_session_by_acp_id(acp_session_id)
        if conversation is None:
            raise ACPSessionError(
                f"Session not found: {acp_session_id}",
                harness=None,
            )

        if (
            conversation.runtime is None
            or conversation.runtime._suspended_permission is None
        ):
            raise ACPSessionError(
                f"Session {acp_session_id} has no pending permission request",
                harness=conversation.harness,
            )

        suspended = conversation.runtime._suspended_permission
        permission_adapter = ACPPermissionAdapter(
            cwd=conversation.cwd,
            require_approval=self.config.require_approval,
        )
        selected_option = permission_adapter.resolve_option_by_id(  # pylint: disable=protected-access
            suspended.options, option_id
        )
        if selected_option is None:
            available = [
                opt.get("optionId") or opt.get("id", "?")
                for opt in suspended.options
                if isinstance(opt, dict)
            ]
            raise ACPSessionError(
                f"Unknown option_id '{option_id}'. "
                f"Available options: {available}",
                harness=conversation.harness,
            )
        permission_result = permission_adapter._selected_result(selected_option)  # pylint: disable=protected-access

        logger.info(
            "ACP resume_permission: session=%s option_id=%s resolved=%s",
            acp_session_id,
            option_id,
            selected_option.get("optionId"),
        )

        async def _handle_event(event: AcpEvent) -> None:
            projected = self._project_event(event)
            for message, last in projected:
                await on_message(message, last)

        async def _resolve_permission(payload: dict[str, Any]) -> dict[str, Any]:
            # For any subsequent permission requests after resume
            next_permission_adapter = ACPPermissionAdapter(
                cwd=conversation.cwd,
                require_approval=self.config.require_approval,
            )
            decision = await next_permission_adapter.resolve_permission(
                session_id="",
                user_id="",
                channel="",
                harness=conversation.harness,
                request_payload=payload,
            )
            return decision.result

        ephemeral = not conversation.keep_session

        try:
            await conversation.runtime.resume_prompt_after_permission(
                chat_id=conversation.chat_id,
                session_id=acp_session_id,
                permission_result=permission_result,
                on_event=_handle_event,
                permission_handler=_resolve_permission,
            )
        finally:
            logger.debug("ACP service: finalizing resumed turn")
            suspended_after = (
                conversation.runtime._suspended_permission is not None
            )

            if suspended_after:
                conversation.suspended_permission = (
                    conversation.runtime._suspended_permission
                )
                await self._store.save(conversation)
            elif ephemeral:
                await conversation.runtime.close()
            else:
                conversation.suspended_permission = None
                await self._store.save(conversation)

        return ACPRunResult(
            harness=conversation.harness,
            session_id=acp_session_id,
            keep_session=conversation.keep_session,
            cwd=conversation.cwd,
            suspended_permission=conversation.runtime._suspended_permission,
        )

    # -----------------------------------------------------------------------
    # Internal Methods
    # -----------------------------------------------------------------------

    async def _get_or_create_conversation(
        self,
        *,
        chat_id: str,
        harness: str,
        cwd: str,
        keep_session: bool,
        existing_session_id: str | None,
        harness_config: ACPHarnessConfig,
    ) -> tuple[ACPConversationSession, bool]:
        """Get an existing conversation or create a new one.

        Returns:
            Tuple of (session, is_ephemeral).
        """
        if keep_session:
            existing = await self._store.get(chat_id, harness)

            if (
                existing is not None
                and existing.runtime is not None
                and existing.runtime.transport.is_running()
            ):
                existing.keep_session = True
                existing.cwd = cwd or existing.cwd
                # Update runtime config with latest settings
                existing.runtime.harness_config = harness_config
                existing.runtime._permission_broker_verified = harness_config.permission_broker_verified
                return existing, False

            runtime = ACPRuntime(harness, harness_config)
            await runtime.start(cwd or ".")
            if existing_session_id:
                acp_session_id = await runtime.load_session(
                    existing_session_id,
                    cwd,
                )
            else:
                acp_session_id = await runtime.new_session(cwd)

            session = ACPConversationSession(
                chat_id=chat_id,
                harness=harness,
                acp_session_id=acp_session_id,
                cwd=cwd,
                keep_session=True,
                capabilities=runtime.capabilities,
                runtime=runtime,
            )
            await self._store.save(session)
            return session, False

        # Ephemeral session
        runtime = ACPRuntime(harness, harness_config)
        await runtime.start(cwd or ".")
        if existing_session_id:
            acp_session_id = await runtime.load_session(
                existing_session_id,
                cwd,
            )
        else:
            acp_session_id = await runtime.new_session(cwd)
        session = ACPConversationSession(
            chat_id=chat_id,
            harness=harness,
            acp_session_id=acp_session_id,
            cwd=cwd,
            keep_session=False,
            capabilities=runtime.capabilities,
            runtime=runtime,
        )
        return session, True

    def _get_harness_config(self, harness: str) -> ACPHarnessConfig:
        """Get harness configuration, raising if invalid."""
        if not self.config.enabled:
            raise ACPConfigurationError(
                "ACP is disabled in config",
                harness=harness,
            )

        harness_config = self.config.harnesses.get(harness)
        if harness_config is None:
            raise ACPConfigurationError(
                f"Unknown ACP harness: {harness}",
                harness=harness,
            )
        if not harness_config.enabled:
            raise ACPConfigurationError(
                f"ACP harness '{harness}' is disabled",
                harness=harness,
            )
        return harness_config

    def _enforce_unverified_harness_policy(
        self,
        *,
        harness: str,
        harness_config: ACPHarnessConfig,
        prompt_blocks: list[dict[str, Any]],
        preapproved: bool = False,
    ) -> None:
        """Enforce policy for unverified harnesses."""
        if not self.config.require_approval:
            return
        if preapproved:
            return
        if harness_config.permission_broker_verified:
            return

        prompt_text = self._extract_prompt_text(prompt_blocks)
        if self._is_obviously_dangerous_prompt(prompt_text):
            raise ACPConfigurationError(
                f"Harness '{harness}' is not verified to request permissions "
                f"before dangerous operations. Please review the prompt and "
                f"use /approve if you trust this operation.",
                harness=harness,
            )

    def _project_event(self, event: AcpEvent) -> list[tuple[Any, bool]]:
        """Project an ACP event into messages.

        Returns a list of (message, is_last) tuples.
        """
        # Simple text projection for now
        # A more sophisticated projector could be added later
        if event.type == "assistant_chunk":
            text = str(event.payload.get("text") or "")
            if not text:
                return []
            # Return as a simple dict that can be converted to a message
            return [({"type": "text", "text": text}, False)]

        if event.type == "tool_start":
            tool_name = event.payload.get("name", "unknown")
            tool_input = event.payload.get("input", {})
            text = f"🔧 Tool: {tool_name}\nInput: {tool_input}"
            return [({"type": "text", "text": text}, True)]

        if event.type == "tool_end":
            tool_name = event.payload.get("name", "unknown")
            output = event.payload.get("output", "")
            if output:
                text = f"✅ Tool result: {tool_name}\n{output}"
                return [({"type": "text", "text": text}, True)]
            return []

        if event.type == "permission_request":
            summary = event.payload.get("summary", {})
            if isinstance(summary, dict):
                text = (
                    f"🔐 Permission Request\n"
                    f"Harness: {summary.get('harness', 'unknown')}\n"
                    f"Tool: {summary.get('tool_name', 'unknown')}\n"
                    f"Kind: {summary.get('tool_kind', 'unknown')}"
                )
            else:
                text = f"🔐 Permission Request: {summary}"
            return [({"type": "text", "text": text}, True)]

        if event.type == "permission_resolved":
            text = "✅ Permission resolved"
            return [({"type": "text", "text": text}, True)]

        if event.type == "error":
            message = event.payload.get("message", "Unknown error")
            text = f"❌ Error: {message}"
            return [({"type": "text", "text": text}, True)]

        if event.type == "run_finished":
            return [({"type": "text", "text": ""}, True)]

        return []

    def _extract_prompt_text(self, prompt_blocks: list[dict[str, Any]]) -> str:
        """Extract text from prompt blocks."""
        texts = []
        for block in prompt_blocks:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif "text" in block:
                    texts.append(block["text"])
        return " ".join(texts)

    def _is_obviously_dangerous_prompt(self, prompt_text: str) -> bool:
        """Check if a prompt is obviously dangerous."""
        dangerous_patterns = [
            r"rm\s+-rf",
            r"delete\s+all",
            r"format\s+disk",
            r"drop\s+table",
            r"truncate\s+table",
            r"exec\s*\(",
            r"eval\s*\(",
            r"system\s*\(",
            r"subprocess",
            r"os\.system",
        ]
        text_lower = prompt_text.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False


# Global service instance
_acp_service: ACPService | None = None


def get_acp_service() -> ACPService | None:
    """Get the global ACP service instance."""
    return _acp_service


def init_acp_service(config: ACPConfig) -> ACPService:
    """Initialize the global ACP service.

    Args:
        config: ACP configuration.

    Returns:
        The initialized service.
    """
    global _acp_service
    _acp_service = ACPService(config=config)
    return _acp_service