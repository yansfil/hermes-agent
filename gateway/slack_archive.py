"""Append-only local archive for Slack message events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


class SlackArchive:
    """Persist Slack messages in one profile-local DuckDB database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS slack_messages (
                    channel_id VARCHAR NOT NULL,
                    message_ts VARCHAR NOT NULL,
                    thread_ts VARCHAR,
                    user_id VARCHAR,
                    message_url VARCHAR NOT NULL,
                    thread_url VARCHAR NOT NULL,
                    received_at TIMESTAMPTZ NOT NULL,
                    event_json JSON NOT NULL,
                    PRIMARY KEY (channel_id, message_ts)
                )
                """
            )

    def record_message(
        self,
        *,
        event: dict[str, Any],
        workspace_url: str,
        received_at: datetime | None = None,
    ) -> None:
        channel_id = str(event["channel"])
        message_ts = str(event["ts"])
        thread_ts = str(event.get("thread_ts") or message_ts)
        workspace_url = workspace_url.rstrip("/") + "/"
        message_url = self._permalink(workspace_url, channel_id, message_ts)
        thread_url = self._thread_permalink(workspace_url, channel_id, thread_ts)
        received_at = received_at or datetime.now(timezone.utc)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO slack_messages (
                    channel_id, message_ts, thread_ts, user_id, message_url,
                    thread_url, received_at, event_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (channel_id, message_ts) DO UPDATE SET
                    thread_ts = excluded.thread_ts,
                    user_id = excluded.user_id,
                    message_url = excluded.message_url,
                    thread_url = excluded.thread_url,
                    received_at = excluded.received_at,
                    event_json = excluded.event_json
                """,
                [
                    channel_id,
                    message_ts,
                    event.get("thread_ts"),
                    event.get("user"),
                    message_url,
                    thread_url,
                    received_at,
                    json.dumps(event, ensure_ascii=False, separators=(",", ":")),
                ],
            )

    def get_message(self, channel_id: str, message_ts: str) -> dict[str, Any]:
        with self._connect(read_only=True) as connection:
            row = connection.execute(
                """
                SELECT channel_id, message_ts, thread_ts, user_id, message_url,
                       thread_url, received_at, event_json
                FROM slack_messages
                WHERE channel_id = ? AND message_ts = ?
                """,
                [channel_id, message_ts],
            ).fetchone()
        if row is None:
            raise KeyError((channel_id, message_ts))
        return {
            "channel_id": row[0],
            "message_ts": row[1],
            "thread_ts": row[2],
            "user_id": row[3],
            "message_url": row[4],
            "thread_url": row[5],
            "received_at": row[6],
            "event": json.loads(row[7]),
        }

    def message_count(self) -> int:
        with self._connect(read_only=True) as connection:
            row = connection.execute("SELECT count(*) FROM slack_messages").fetchone()
        if row is None:
            return 0
        return int(row[0])

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(str(self.path), read_only=read_only)

    @staticmethod
    def _permalink(workspace_url: str, channel_id: str, message_ts: str) -> str:
        return f"{workspace_url}archives/{channel_id}/p{message_ts.replace('.', '')}"

    @staticmethod
    def _thread_permalink(workspace_url: str, channel_id: str, thread_ts: str) -> str:
        root_url = SlackArchive._permalink(workspace_url, channel_id, thread_ts)
        return f"{root_url}?thread_ts={thread_ts}&cid={channel_id}"
