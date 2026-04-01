# -*- coding: utf-8 -*-
"""Session storage for ACP conversations.

This module provides in-memory storage for active ACP sessions,
enabling session persistence across multiple tool calls within the same chat.

Sessions are keyed by (chat_id, harness) to allow different harnesses
to have independent sessions within the same chat.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .types import ACPConversationSession, utc_now

logger = logging.getLogger(__name__)


class ACPSessionStore:
    """Store active ACP runtime state keyed by chat and harness.

    This is an in-memory store with optional persistence to disk.
    Sessions are keyed by (chat_id, harness) tuple.

    Attributes:
        sessions: Dictionary of active sessions by (chat_id, harness).
    """

    def __init__(self, save_dir: str | None = None):
        """Initialize the session store.

        Args:
            save_dir: Optional directory for session persistence.
        """
        self._lock = asyncio.Lock()
        self._sessions: dict[tuple[str, str], ACPConversationSession] = {}
        self._save_dir = Path(save_dir).expanduser() if save_dir else None

    async def get(
        self,
        chat_id: str,
        harness: str,
    ) -> ACPConversationSession | None:
        """Get an active session by chat_id and harness.

        Args:
            chat_id: The chat session identifier.
            harness: The harness name.

        Returns:
            The session if found and active, None otherwise.
        """
        async with self._lock:
            return self._sessions.get((chat_id, harness))

    async def save(self, session: ACPConversationSession) -> None:
        """Save a session to the store.

        Args:
            session: The session to save.
        """
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
        """Delete a session from the store.

        Args:
            chat_id: The chat session identifier.
            harness: The harness name.

        Returns:
            The deleted session if found, None otherwise.
        """
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
        """List all sessions, optionally filtered.

        Args:
            chat_id: Optional filter by chat_id.
            harness: Optional filter by harness.

        Returns:
            List of session info dictionaries.
        """
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
        """Get a session by its ACP session ID.

        Args:
            acp_session_id: The ACP protocol session ID.

        Returns:
            The session if found, None otherwise.
        """
        async with self._lock:
            for session in self._sessions.values():
                if session.acp_session_id == acp_session_id:
                    return session
            return None

    async def clear_inactive(self, max_age_seconds: float = 3600.0) -> int:
        """Clear inactive sessions that haven't been updated recently.

        Args:
            max_age_seconds: Maximum age in seconds for inactive sessions.

        Returns:
            Number of sessions cleared.
        """
        now = utc_now()
        cleared = 0
        async with self._lock:
            to_remove = []
            for key, session in self._sessions.items():
                age = (now - session.updated_at).total_seconds()
                if age > max_age_seconds:
                    # Close runtime if present
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

    async def persist_to_disk(self) -> None:
        """Persist session metadata to disk (without runtime references)."""
        if self._save_dir is None:
            return

        self._save_dir.mkdir(parents=True, exist_ok=True)

        async with self._lock:
            for session in self._sessions.values():
                if not session.keep_session:
                    continue
                session_file = self._save_dir / f"{session.acp_session_id}.json"
                try:
                    data = {
                        "chat_id": session.chat_id,
                        "harness": session.harness,
                        "acp_session_id": session.acp_session_id,
                        "cwd": session.cwd,
                        "keep_session": session.keep_session,
                        "capabilities": session.capabilities,
                        "updated_at": session.updated_at.isoformat(),
                    }
                    session_file.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to persist session %s: %s",
                        session.acp_session_id,
                        e,
                    )

    async def load_from_disk(self) -> int:
        """Load persisted session metadata from disk.

        Note: This only loads metadata; the runtime must be recreated
        when the session is resumed.

        Returns:
            Number of sessions loaded.
        """
        if self._save_dir is None or not self._save_dir.exists():
            return 0

        loaded = 0
        async with self._lock:
            for session_file in self._save_dir.glob("*.json"):
                try:
                    data = json.loads(session_file.read_text(encoding="utf-8"))
                    session = ACPConversationSession(
                        chat_id=data["chat_id"],
                        harness=data["harness"],
                        acp_session_id=data["acp_session_id"],
                        cwd=data["cwd"],
                        keep_session=data.get("keep_session", True),
                        capabilities=data.get("capabilities", {}),
                        updated_at=datetime.fromisoformat(data["updated_at"]),
                    )
                    self._sessions[(session.chat_id, session.harness)] = session
                    loaded += 1
                except Exception as e:
                    logger.warning(
                        "Failed to load session from %s: %s",
                        session_file,
                        e,
                    )

        if loaded > 0:
            logger.info("Loaded %d persisted ACP sessions", loaded)
        return loaded


# Global session store instance
_session_store: ACPSessionStore | None = None


def get_session_store() -> ACPSessionStore:
    """Get the global session store instance."""
    global _session_store
    if _session_store is None:
        _session_store = ACPSessionStore()
    return _session_store


def init_session_store(save_dir: str | None = None) -> ACPSessionStore:
    """Initialize the global session store with optional persistence.

    Args:
        save_dir: Optional directory for session persistence.

    Returns:
        The initialized session store.
    """
    global _session_store
    _session_store = ACPSessionStore(save_dir=save_dir)
    return _session_store