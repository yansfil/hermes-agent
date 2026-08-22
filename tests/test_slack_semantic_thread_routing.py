"""Gating tests for Slack semantic thread routing and free-response scoping.

Covers the fork-local behavior layered onto the upstream adapter:
free-response channels admit only top-level messages, and unmentioned
replies inside in-scope threads pass through ignore prefixes, the
deterministic route, and (when enabled) the semantic classifier.
"""

import asyncio

from agent.semantic_router import RoutingDecision, ThreadMessage
from gateway.config import PlatformConfig, _normalize_semantic_thread_routing
from plugins.platforms.slack.adapter import SlackAdapter, _DeterministicThreadRoute


def run(coro):
    return asyncio.run(coro)


def make_adapter(extra=None):
    config = PlatformConfig(extra=extra or {})
    adapter = SlackAdapter(config)
    adapter._bot_user_id = "UBOT"
    adapter._team_bot_user_ids["T1"] = "UBOT"
    adapter._has_active_session_for_thread = lambda **_: False

    async def no_thread_context(**_):
        return ""

    async def no_parent_text(**_):
        return ""

    async def user_name(*_, **__):
        return "Hoyeon"

    async def no_semantic_context(*_, **__):
        return ()

    adapter._fetch_thread_context = no_thread_context
    adapter._fetch_thread_parent_text = no_parent_text
    adapter._resolve_user_name = user_name
    adapter._fetch_semantic_thread_context = no_semantic_context
    return adapter


def capture_handled(adapter):
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture
    return handled


def slack_event(text, ts="200.000", thread_ts=None):
    event = {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "team": "T1",
        "user": "U123",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


FREE_RESPONSE = {
    "allowed_channels": ["C123"],
    "free_response_channels": ["C123"],
    "reply_in_thread": True,
}


def in_scope(adapter, thread_ts="100.000"):
    adapter._bot_message_ts.add(thread_ts)
    return thread_ts


# --- free-response scoping -------------------------------------------------


def test_free_response_processes_top_level_without_mention():
    adapter = make_adapter(dict(FREE_RESPONSE))
    handled = capture_handled(adapter)

    run(adapter._handle_slack_message(slack_event("서버 배포해줘")))

    assert len(handled) == 1
    assert handled[0].text == "서버 배포해줘"


def test_free_response_drops_thread_reply_outside_bot_threads():
    adapter = make_adapter(dict(FREE_RESPONSE))
    handled = capture_handled(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("이건 사람들끼리 얘기", thread_ts="100.000")
        )
    )

    assert handled == []


def test_in_scope_thread_reply_with_direct_address_processes():
    adapter = make_adapter(dict(FREE_RESPONSE))
    handled = capture_handled(adapter)
    thread_ts = in_scope(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("모닥아 방금 내용 정리해줘", thread_ts=thread_ts)
        )
    )

    assert len(handled) == 1


def test_in_scope_thread_reply_with_ambiguous_text_stays_silent():
    adapter = make_adapter(dict(FREE_RESPONSE))
    handled = capture_handled(adapter)
    thread_ts = in_scope(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("점심 뭐 먹을까", thread_ts=thread_ts)
        )
    )

    assert handled == []


def test_ignore_prefix_silences_in_scope_thread_reply():
    adapter = make_adapter(dict(FREE_RESPONSE))
    handled = capture_handled(adapter)
    thread_ts = in_scope(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("xx 모닥아 이건 무시해", thread_ts=thread_ts)
        )
    )

    assert handled == []


def test_mention_pattern_acts_as_wake_word_for_top_level():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "mention_patterns": ["모닥아"],
            "reply_in_thread": True,
        }
    )
    handled = capture_handled(adapter)

    run(adapter._handle_slack_message(slack_event("모닥아 이 링크 읽어줘")))

    assert len(handled) == 1


# --- semantic routing ------------------------------------------------------


SEMANTIC = {
    "enabled": True,
    "mode": "enforce",
    "provider": "openai-codex",
    "model": "gpt-5.4-mini",
    "confidence_threshold": 0.85,
    "timeout_seconds": 3,
}


def make_semantic_adapter(decision):
    adapter = make_adapter(
        dict(FREE_RESPONSE, semantic_thread_routing=dict(SEMANTIC))
    )
    calls = []

    async def router(**kwargs):
        calls.append(kwargs)
        if isinstance(decision, Exception):
            raise decision
        return decision

    adapter._semantic_thread_router = router
    return adapter, calls


def test_semantic_enforce_respond_processes():
    adapter, calls = make_semantic_adapter(
        RoutingDecision("respond", 0.97, "asked the bot")
    )
    handled = capture_handled(adapter)
    thread_ts = in_scope(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("그 부분 다시 설명해줄래?", thread_ts=thread_ts)
        )
    )

    assert len(handled) == 1
    assert len(calls) == 1
    assert isinstance(calls[0]["current_message"], ThreadMessage)


def test_semantic_enforce_ignore_stays_silent():
    adapter, _ = make_semantic_adapter(
        RoutingDecision("ignore", 0.99, "human chatter")
    )
    handled = capture_handled(adapter)
    thread_ts = in_scope(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("우리끼리 회의 잡자", thread_ts=thread_ts)
        )
    )

    assert handled == []


def test_semantic_low_confidence_stays_silent():
    adapter, _ = make_semantic_adapter(
        RoutingDecision("respond", 0.5, "maybe")
    )
    handled = capture_handled(adapter)
    thread_ts = in_scope(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("흠 그런가", thread_ts=thread_ts)
        )
    )

    assert handled == []


def test_semantic_router_failure_fails_closed():
    adapter, _ = make_semantic_adapter(RuntimeError("provider down"))
    handled = capture_handled(adapter)
    thread_ts = in_scope(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("이거 처리해줘", thread_ts=thread_ts)
        )
    )

    assert handled == []


def test_semantic_shadow_mode_never_dispatches_on_router_alone():
    adapter, calls = make_semantic_adapter(
        RoutingDecision("respond", 0.99, "asked the bot")
    )
    adapter.config.extra["semantic_thread_routing"]["mode"] = "shadow"
    handled = capture_handled(adapter)
    thread_ts = in_scope(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("그 부분 다시 설명해줘", thread_ts=thread_ts)
        )
    )

    assert handled == []
    assert len(calls) == 1  # shadow mode still observes


def test_mention_bypasses_semantic_gate():
    adapter, calls = make_semantic_adapter(
        RoutingDecision("ignore", 0.99, "would have ignored")
    )
    handled = capture_handled(adapter)
    thread_ts = in_scope(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("<@UBOT> 이건 무조건 답해줘", thread_ts=thread_ts)
        )
    )

    assert len(handled) == 1
    assert calls == []


# --- deterministic route and normalizer ------------------------------------


def test_deterministic_route_classification():
    adapter = make_adapter()
    route = adapter._slack_deterministic_thread_route

    assert route("모닥아 정리해줘", "UBOT") is _DeterministicThreadRoute.RESPOND
    assert route("모닥이에게 시켜볼까요?", "UBOT") is _DeterministicThreadRoute.IGNORE
    assert route("xx 무시해줘", "UBOT") is _DeterministicThreadRoute.IGNORE
    assert route("호연님 이건 어때요", "UBOT") is _DeterministicThreadRoute.IGNORE
    assert route("점심 뭐 먹지", "UBOT") is _DeterministicThreadRoute.AMBIGUOUS


def test_normalizer_defaults_and_clamps():
    disabled = _normalize_semantic_thread_routing(None)
    assert disabled["enabled"] is False

    clamped = _normalize_semantic_thread_routing(
        {
            "enabled": True,
            "mode": "enforce",
            "provider": "openai-codex",
            "model": "gpt-5.4-mini",
            "context_messages": 99,
            "confidence_threshold": 7,
            "timeout_seconds": 1000,
        }
    )
    assert clamped["context_messages"] == 12
    assert clamped["confidence_threshold"] == 1.0
    assert clamped["timeout_seconds"] == 15.0

    blank_model = _normalize_semantic_thread_routing(
        {"enabled": True, "mode": "enforce", "provider": "openai-codex", "model": " "}
    )
    assert blank_model["enabled"] is False
