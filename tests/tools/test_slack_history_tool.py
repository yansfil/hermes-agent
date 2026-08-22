import json

import pytest

from tools.slack_history_tool import slack_history


class FakeSlackClient:
    def __init__(self):
        self.calls = []

    def conversations_list(self, **kwargs):
        self.calls.append(("conversations_list", kwargs))
        return {
            "channels": [
                {"id": "C123", "name": "core", "is_private": False, "purpose": {"value": "Company core"}},
            ],
            "response_metadata": {"next_cursor": ""},
        }

    def users_list(self, **kwargs):
        self.calls.append(("users_list", kwargs))
        return {
            "members": [{"id": "U_KYLE", "real_name": "Kyle", "name": "kyle", "is_bot": False}],
            "response_metadata": {"next_cursor": ""},
        }

    def conversations_history(self, **kwargs):
        self.calls.append(("conversations_history", kwargs))
        return {
            "messages": [
                {"ts": "1785680000.000001", "user": "U_KYLE", "text": "강의 제작 진행"},
            ],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }

    def conversations_replies(self, **kwargs):
        self.calls.append(("conversations_replies", kwargs))
        return {
            "messages": [
                {"ts": kwargs["ts"], "user": "U_KYLE", "text": "원글"},
                {"ts": "1785680001.000001", "user": "U_GRAB", "text": "답글", "thread_ts": kwargs["ts"]},
            ],
            "response_metadata": {"next_cursor": ""},
        }


def test_list_channels_excludes_private_and_direct_messages():
    client = FakeSlackClient()

    result = json.loads(slack_history(action="list_channels", client=client))

    assert result["channels"] == [
        {"id": "C123", "name": "core", "purpose": "Company core"},
    ]
    assert client.calls == [("conversations_list", {"types": "public_channel", "limit": 200})]


def test_list_users_returns_human_identity_map():
    client = FakeSlackClient()

    result = json.loads(slack_history(action="list_users", client=client))

    assert result["users"] == [{"id": "U_KYLE", "name": "kyle", "real_name": "Kyle"}]
    assert client.calls == [("users_list", {"limit": 200})]


def test_read_messages_requires_explicit_channel_and_returns_bounded_page():
    client = FakeSlackClient()

    result = json.loads(slack_history(action="read_messages", channel="C123", limit=1, client=client))

    assert result["messages"] == [
        {"ts": "1785680000.000001", "user_id": "U_KYLE", "text": "강의 제작 진행", "thread_ts": None},
    ]
    assert client.calls == [
        ("conversations_history", {"channel": "C123", "limit": 1}),
    ]


def test_read_messages_rejects_missing_channel():
    with pytest.raises(ValueError, match="channel is required"):
        slack_history(action="read_messages", client=FakeSlackClient())


def test_registry_argument_mapping_is_accepted():
    client = FakeSlackClient()

    result = json.loads(slack_history({"action": "read_messages", "channel": "C123", "limit": 1}, client=client))

    assert result["channel"] == "C123"
    assert client.calls == [("conversations_history", {"channel": "C123", "limit": 1})]


def test_read_thread_requires_public_channel_and_returns_replies():
    client = FakeSlackClient()

    result = json.loads(slack_history({"action": "read_thread", "channel": "C123", "thread_ts": "1785680000.000001"}, client=client))

    assert [message["text"] for message in result["messages"]] == ["원글", "답글"]
    assert client.calls == [
        ("conversations_replies", {"channel": "C123", "ts": "1785680000.000001", "limit": 100}),
    ]
