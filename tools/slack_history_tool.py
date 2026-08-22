"""Read-only Slack public-channel history for scoped workspace analysis.

The tool deliberately excludes DMs, group DMs, and private channels.
Every message read requires an explicit public channel ID supplied by the agent.
"""

import json
import os
from typing import Any, Optional

from tools.registry import registry


MAX_PAGE_SIZE = 200


SLACK_HISTORY_SCHEMA = {
    "name": "slack_history",
    "description": (
        "Read-only Slack public-channel history. Use list_channels first, then "
        "read_messages or read_thread with an explicit channel ID. DMs and private "
        "channels are intentionally unavailable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_channels", "list_users", "read_messages", "read_thread"],
                "description": "Read-only action to perform.",
            },
            "channel": {
                "type": "string",
                "description": "Required for read_messages/read_thread. Public Slack channel ID from list_channels.",
            },
            "thread_ts": {
                "type": "string",
                "description": "Required for read_thread. Parent Slack message timestamp.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": "Maximum records to return. Default 100.",
            },
            "cursor": {
                "type": "string",
                "description": "Pagination cursor returned by a previous call.",
            },
            "oldest": {
                "type": "string",
                "description": "Optional inclusive Slack timestamp lower bound for read_messages.",
            },
            "latest": {
                "type": "string",
                "description": "Optional exclusive Slack timestamp upper bound for read_messages.",
            },
        },
        "required": ["action"],
    },
}


def check_slack_history_requirements() -> bool:
    """Expose the tool only when a Slack bot token and SDK are available."""
    if not os.getenv("SLACK_BOT_TOKEN", "").strip():
        return False
    try:
        import slack_sdk  # noqa: F401
    except ImportError:
        return False
    return True


def _client_from_environment() -> Any:
    from slack_sdk import WebClient

    return WebClient(token=os.environ["SLACK_BOT_TOKEN"])


def _page_size(limit: Optional[int]) -> int:
    if limit is None:
        return 100
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be an integer from 1 to {MAX_PAGE_SIZE}")
    return limit


def _cursor_payload(response: Any) -> dict:
    metadata = response.get("response_metadata", {}) if hasattr(response, "get") else {}
    return {"next_cursor": metadata.get("next_cursor", ""), "has_more": bool(response.get("has_more", False))}


def _public_channels(response: Any) -> list[dict]:
    return [
        {
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "purpose": item.get("purpose", {}).get("value", ""),
        }
        for item in response.get("channels", [])
        if not item.get("is_private", False) and not item.get("is_im", False) and not item.get("is_mpim", False)
    ]


def _human_users(response: Any) -> list[dict]:
    return [
        {"id": item.get("id", ""), "name": item.get("name", ""), "real_name": item.get("real_name", "")}
        for item in response.get("members", [])
        if not item.get("is_bot", False) and not item.get("deleted", False)
    ]


def _messages(response: Any) -> list[dict]:
    return [
        {
            "ts": item.get("ts", ""),
            "user_id": item.get("user", ""),
            "text": item.get("text", ""),
            "thread_ts": item.get("thread_ts"),
        }
        for item in response.get("messages", [])
        if item.get("type", "message") == "message"
    ]


def slack_history(
    args: Optional[dict[str, Any]] = None,
    *,
    client: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """Execute a bounded, read-only public-channel Slack history request.

    Registry handlers receive the model arguments as one mapping.  Accepting
    keyword arguments too keeps this function straightforward to unit test.
    """
    if args is not None and not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    payload = dict(args or {})
    payload.update(kwargs)

    action = payload.get("action")
    channel = payload.get("channel")
    thread_ts = payload.get("thread_ts")
    limit = payload.get("limit")
    cursor = payload.get("cursor")
    oldest = payload.get("oldest")
    latest = payload.get("latest")

    client = client or _client_from_environment()
    page_size = _page_size(limit)

    if action == "list_channels":
        response = client.conversations_list(types="public_channel", limit=MAX_PAGE_SIZE, **({"cursor": cursor} if cursor else {}))
        return json.dumps({"channels": _public_channels(response), **_cursor_payload(response)}, ensure_ascii=False)

    if action == "list_users":
        response = client.users_list(limit=MAX_PAGE_SIZE, **({"cursor": cursor} if cursor else {}))
        return json.dumps({"users": _human_users(response), **_cursor_payload(response)}, ensure_ascii=False)

    if action == "read_messages":
        if not channel:
            raise ValueError("channel is required for read_messages")
        request_args: dict[str, Any] = {"channel": channel, "limit": page_size}
        if cursor:
            request_args["cursor"] = cursor
        if oldest:
            request_args["oldest"] = oldest
        if latest:
            request_args["latest"] = latest
        response = client.conversations_history(**request_args)
        return json.dumps({"channel": channel, "messages": _messages(response), **_cursor_payload(response)}, ensure_ascii=False)

    if action == "read_thread":
        if not channel:
            raise ValueError("channel is required for read_thread")
        if not thread_ts:
            raise ValueError("thread_ts is required for read_thread")
        request_args = {"channel": channel, "ts": thread_ts, "limit": page_size}
        if cursor:
            request_args["cursor"] = cursor
        response = client.conversations_replies(**request_args)
        return json.dumps({"channel": channel, "thread_ts": thread_ts, "messages": _messages(response), **_cursor_payload(response)}, ensure_ascii=False)

    raise ValueError("action must be list_channels, list_users, read_messages, or read_thread")


registry.register(
    name="slack_history",
    toolset="slack",
    schema=SLACK_HISTORY_SCHEMA,
    handler=slack_history,
    check_fn=check_slack_history_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    is_async=False,
    description="Read-only public Slack channel and message history.",
    emoji="💬",
    max_result_size_chars=50_000,
)
