from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter
from gateway.slack_archive import SlackArchive


def make_adapter():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test"))
    adapter._bot_user_id = "U_BOT"
    adapter._has_active_session_for_thread = lambda **_: False

    async def no_parent_text(**_):
        return ""

    adapter._fetch_thread_parent_text = no_parent_text
    return adapter


def test_records_message_with_message_and_thread_permalinks(tmp_path):
    archive = SlackArchive(tmp_path / "slack-archive.duckdb")

    archive.record_message(
        event={
            "channel": "C123",
            "ts": "1787028698.717029",
            "thread_ts": "1787028000.000001",
            "user": "U_KYLE",
            "text": "DuckDB에 저장하자",
        },
        workspace_url="https://modakbul.slack.com/",
        received_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
    )

    message = archive.get_message("C123", "1787028698.717029")

    assert message["channel_id"] == "C123"
    assert message["thread_ts"] == "1787028000.000001"
    assert message["message_url"] == "https://modakbul.slack.com/archives/C123/p1787028698717029"
    assert message["thread_url"] == "https://modakbul.slack.com/archives/C123/p1787028000000001?thread_ts=1787028000.000001&cid=C123"
    assert message["event"]["text"] == "DuckDB에 저장하자"


def test_upserts_redelivered_message_without_duplicate(tmp_path):
    archive = SlackArchive(tmp_path / "slack-archive.duckdb")
    original = {"channel": "C123", "ts": "1787028698.717029", "user": "U_KYLE", "text": "원문"}
    updated = {"channel": "C123", "ts": "1787028698.717029", "user": "U_KYLE", "text": "수정된 원문"}

    archive.record_message(event=original, workspace_url="https://modakbul.slack.com/")
    archive.record_message(event=updated, workspace_url="https://modakbul.slack.com/")

    assert archive.message_count() == 1
    assert archive.get_message("C123", "1787028698.717029")["event"]["text"] == "수정된 원문"


@pytest.mark.asyncio
async def test_adapter_archives_public_message_before_mention_filtering():
    adapter = make_adapter()
    archive = MagicMock()
    event = {
        "channel": "C123",
        "channel_type": "channel",
        "ts": "1787028698.717029",
        "user": "U_KYLE",
        "text": "일반 대화",
    }
    adapter._slack_archive = archive
    adapter._team_workspace_urls = {"": "https://modakbul.slack.com/"}

    # The unmentioned message is dropped by gating, but the archive write
    # must land first — that ordering is the feature.
    await adapter._handle_slack_message(event)

    archive.record_message.assert_called_once_with(
        event=event,
        workspace_url="https://modakbul.slack.com/",
    )


@pytest.mark.parametrize(
    "event",
    [
        {"channel": "D123", "channel_type": "im", "ts": "1.1", "text": "DM"},
        {"channel": "G123", "channel_type": "group", "ts": "1.2", "text": "비공개"},
        {"channel": "C123", "channel_type": "channel", "ts": "1.3", "subtype": "message_changed"},
        {"channel": "C123", "channel_type": "channel", "ts": "1.4", "subtype": "message_deleted"},
    ],
)
def test_adapter_never_archives_non_public_or_message_state_events(event):
    adapter = make_adapter()
    archive = MagicMock()
    adapter._slack_archive = archive
    adapter._team_workspace_urls = {"": "https://modakbul.slack.com/"}

    adapter._archive_slack_message(event)

    archive.record_message.assert_not_called()


def test_archive_disabled_without_configured_path(monkeypatch):
    monkeypatch.delenv("SLACK_ARCHIVE_DUCKDB_PATH", raising=False)
    adapter = make_adapter()

    adapter._configure_slack_archive()

    assert adapter._slack_archive is None


def test_archive_write_failure_never_breaks_message_handling():
    adapter = make_adapter()
    archive = MagicMock()
    archive.record_message.side_effect = RuntimeError("disk full")
    adapter._slack_archive = archive
    adapter._team_workspace_urls = {"": "https://modakbul.slack.com/"}

    adapter._archive_slack_message(
        {"channel": "C123", "channel_type": "channel", "ts": "9.9", "text": "x"}
    )  # must not raise
