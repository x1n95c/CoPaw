# -*- coding: utf-8 -*-
"""Session storage for ACP conversations.

This module provides in-memory storage for active ACP sessions,
enabling session persistence across multiple tool calls within the same chat.

Sessions are keyed by (chat_id, harness) to allow different harnesses
to have independent sessions within the same chat.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from .types import ACPConversationSession, utc_now

logger = logging.getLogger(__name__)


class ACPSessionStore:
    """Store active ACP runtime state keyed by chat and harness."""

    def __init__(self, save_dir: str | None = None):
        """Initialize the session store.

        Args:
            save_dir: Deprecated compatibility argument. Session storage is
                in-memory only.
        """
        self._lock = asyncio.Lock()
        self._sessions: dict[tuple[str, str], ACPConversationSession] = {}

    async def get(
        self,
        chat_id: str,
        harness: str,
    ) -> ACPConversationSession | None:
        """Get an active session by chat_id and harness."""
        async with self._lock:
            return self._sessions.get((chat_id, harness))

    async def save(self, session: ACPConversationSession) -> None:
        """Save a session to the store."""
        async with self._lock:
            session.updated_at = utc_now()
            self._sessions[(session.chat_id, session.harness)] = session
            logger.debug(
                "Saved ACP session: chat_id=%s, harness=%s, acp_session_id=%s",
                session.chat_id,
                session.harness,
                session.acp_session_id,
            )

    async def delete(
        self,
        chat_id: str,
        harness: str,
    ) -> ACPConversationSession | None:
        """Delete a session from the store."""
        async with self._lock:
            session = self._sessions.pop((chat_id, harness), None)
            if session is not None:
                logger.debug(
                    "Deleted ACP session: chat_id=%s, harness=%s",
                    chat_id,
                    harness,
                )
            return session

    async def list_sessions(
        self,
        chat_id: Optional[str] = None,
        harness: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List all sessions, optionally filtered."""
        async with self._lock:
            result = []
            for (cid, hname), session in self._sessions.items():
                if chat_id and cid != chat_id:
                    continue
                if harness and hname != harness:
                    continue
                result.append({
                    "chat_id": session.chat_id,
                    "harness": session.harness,
                    "acp_session_id": session.acp_session_id,
                    "cwd": session.cwd,
                    "keep_session": session.keep_session,
                    "updated_at": session.updated_at.isoformat(),
                    "has_active_runtime": (
                        session.runtime is not None
                        and session.runtime.transport.is_running()
                    ),
                })
            return result

    async def get_session_by_acp_id(
        self,
        acp_session_id: str,
    ) -> ACPConversationSession | None:
        """Get a session by its ACP session ID."""
        async with self._lock:
            for session in self._sessions.values():
                if session.acp_session_id == acp_session_id:
                    return session
            return None

    async def clear_inactive(self, max_age_seconds: float = 3600.0) -> int:
        """Clear inactive sessions that haven't been updated recently."""
        now = utc_now()
        cleared = 0
        async with self._lock:
            to_remove = []
            for key, session in self._sessions.items():
                age = (now - session.updated_at).total_seconds()
                if age > max_age_seconds:
                    if session.runtime is not None:
                        try:
                            await session.runtime.close()
                        except Exception as e:
                            logger.warning(
                                "Error closing inactive session runtime: %s",
                                e,
                            )
                    to_remove.append(key)
                    cleared += 1

            for key in to_remove:
                del self._sessions[key]

        if cleared > 0:
            logger.info("Cleared %d inactive ACP sessions", cleared)
        return cleared

