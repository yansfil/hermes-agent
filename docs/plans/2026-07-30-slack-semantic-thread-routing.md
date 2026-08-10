# Slack Semantic Thread Routing Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Let Modakie respond naturally to unmentioned follow-up requests in Slack threads while remaining silent during human-to-human discussion.

**Architecture:** Preserve explicit @mentions and leading wake words as unconditional routing signals.
For unmentioned messages in a thread where the bot has already participated, keep deterministic ignore signals and then call a small structured-output classifier only for ambiguous messages.
The classifier uses `openai-codex/gpt-5.4-mini`, returns `respond`, `ignore`, or `uncertain`, and fails closed to silence.

**Tech Stack:** Python 3.11, Hermes gateway Slack adapter, Slack Web API thread replies, `agent.auxiliary_client.async_call_llm`, pytest, OpenAI Codex OAuth.

**Current baseline:** Preserve the existing uncommitted wake-word and regex smart-thread-reply work in `gateway/platforms/slack.py`, `gateway/config.py`, `tests/gateway/test_slack.py`, and `tests/gateway/test_slack_mention.py`.
Do not turn `slack.require_mention` off and do not add the channel to `free_response_channels`.

---

## Product policy

- An explicit `@modakiee` mention or a leading configured wake word such as `모닥아` always reaches the main agent.
- A new top-level channel message without an explicit invocation remains silent.
- Only an unmentioned reply inside a bot-participated Slack thread is eligible for semantic classification.
- Obvious human-directed speech, third-person references to Modakie, and configured ignore prefixes remain silent without calling a model.
- `uncertain`, parsing failure, rate limit, provider failure, and timeout all mean silence.
- The product preference is high precision over recall: it is better to require `모닥아` once than to interrupt humans.

## Target runtime flow

```text
Slack message event
  |
  +-- direct @mention or leading wake word?
  |     -> strip invocation prefix
  |     -> main Hermes agent
  |
  +-- top-level channel message without invocation?
  |     -> silent
  |
  +-- unmentioned reply in a bot-participated thread?
        |
        +-- ignore prefix / direct human address / third-person bot reference?
        |     -> silent
        |
        +-- deterministic direct follow-up pattern?
        |     -> main Hermes agent
        |
        +-- otherwise fetch recent thread context (max 8 messages)
              -> gpt-5.4-mini semantic router
                   |
                   +-- respond with confidence >= 0.85
                   |     -> main Hermes agent
                   |
                   +-- ignore / uncertain / error / timeout
                         -> silent
```

## Configuration contract

```yaml
slack:
  require_mention: true
  wake_words:
    - 모닥아
  smart_thread_replies: true
  semantic_thread_routing:
    enabled: true
    mode: shadow # off | shadow | enforce
    provider: openai-codex
    model: gpt-5.4-mini
    context_messages: 8
    confidence_threshold: 0.85
    timeout_seconds: 3
```

`off` keeps existing deterministic routing only.
`shadow` records semantic decisions but preserves the existing deterministic routing decision.
`enforce` permits an ambiguous unmentioned message only when the semantic router returns `respond` with confidence at or above the configured threshold.

## Task 1: Add the semantic router domain module

**Objective:** Define a small, independently testable API that turns thread context into a safe routing decision.

**Files:**
- Create: `agent/semantic_router.py`
- Create: `tests/agent/test_semantic_router.py`

**Step 1: Write failing tests.**

Test these externally visible behaviors before implementation:

```python
async def test_returns_respond_for_valid_structured_output(): ...
async def test_returns_uncertain_for_invalid_json(): ...
async def test_returns_uncertain_for_unknown_decision(): ...
async def test_returns_uncertain_when_confidence_is_out_of_range(): ...
async def test_returns_uncertain_when_llm_call_times_out(): ...
async def test_uses_configured_codex_mini_model(): ...
```

Inject an async LLM caller into the router rather than patching global networking.
The fake caller must assert that the request uses `provider="openai-codex"`, `model="gpt-5.4-mini"`, `temperature=0`, and `max_tokens<=100`.

**Step 2: Run RED.**

Run:

```bash
pytest tests/agent/test_semantic_router.py -v
```

Expected: FAIL because `agent.semantic_router` does not exist.

**Step 3: Implement the minimum API.**

Define immutable data types:

```python
@dataclass(frozen=True)
class ThreadMessage:
    author: str
    is_bot: bool
    text: str

@dataclass(frozen=True)
class RoutingDecision:
    decision: Literal["respond", "ignore", "uncertain"]
    confidence: float
    reason: str
```

Define:

```python
async def classify_thread_message(
    *,
    current_message: ThreadMessage,
    recent_messages: Sequence[ThreadMessage],
    bot_name: str,
    provider: str,
    model: str,
    timeout_seconds: float,
    llm_caller: Callable[..., Awaitable[Any]] = async_call_llm,
) -> RoutingDecision:
    ...
```

Call `agent.auxiliary_client.async_call_llm` with the explicit provider and model.
Do not rely on the main model or any existing unrelated `auxiliary.*` task slot.
Pass a two-message prompt with an instruction and a JSON context payload.
Parse JSON defensively and return `uncertain` for every malformed or failed result.

**Step 4: Run GREEN.**

Run:

```bash
pytest tests/agent/test_semantic_router.py -v
```

Expected: PASS.

**Step 5: Refactor.**

Keep the prompt constructor and parser private to this module.
Do not expose Slack-specific APIs from this module.

## Task 2: Specify and test the classifier prompt contract

**Objective:** Make the routing policy explicit and prevent the classifier from replying conversationally.

**Files:**
- Modify: `agent/semantic_router.py`
- Modify: `tests/agent/test_semantic_router.py`

**Step 1: Write failing prompt-contract tests.**

Verify the classifier prompt:

- identifies itself as a routing classifier, not a chat assistant;
- contains only the last 8 messages and the current message;
- marks bot and human speakers distinctly;
- says third-person bot references are `ignore`;
- says ambiguous messages are `uncertain`;
- requires JSON only.

Use Korean examples:

```text
모닥이에게 시켜볼까요? -> ignore
카일님은 어떻게 생각하세요? -> ignore
그럼 방금 기준으로 다시 정리해줘. -> respond when the bot supplied the referenced basis
그럼 그 기준으로 가자. -> uncertain
```

**Step 2: Run RED.**

Run:

```bash
pytest tests/agent/test_semantic_router.py -v
```

Expected: FAIL on the missing prompt assertions.

**Step 3: Implement the fixed prompt.**

The model output schema is exactly:

```json
{"decision":"respond|ignore|uncertain","confidence":0.0,"reason":"20자 이내 한국어"}
```

Tell the model that any uncertainty must be `uncertain`.
Tell the model that unsolicited interruption is worse than a false negative.

**Step 4: Run GREEN.**

Run:

```bash
pytest tests/agent/test_semantic_router.py -v
```

Expected: PASS.

## Task 3: Add Slack semantic-router configuration

**Objective:** Make the feature opt-in and prevent model, threshold, and rollout mode from being hard-coded.

**Files:**
- Modify: `hermes_cli/config.py`
- Modify: `gateway/config.py`
- Modify: `tests/gateway/test_slack_mention.py`

**Step 1: Write failing config bridge tests.**

Add a YAML fixture with the `slack.semantic_thread_routing` object.
Assert that `GatewayConfig.platforms[Platform.SLACK].extra` preserves the object without converting numeric fields to strings.
Assert that an omitted block causes the semantic router to be disabled.

**Step 2: Run RED.**

Run:

```bash
pytest tests/gateway/test_slack_mention.py -v
```

Expected: FAIL because the nested Slack setting is not bridged.

**Step 3: Implement schema defaults and bridge.**

Set defaults to disabled, shadow mode, `openai-codex`, `gpt-5.4-mini`, eight context messages, 0.85 threshold, and three-second timeout.
Validate `mode`, clamp `context_messages` to 3 through 12, clamp the threshold to 0 through 1, and reject blank provider or model values by disabling the feature.
Bridge the nested object directly into the Slack platform `extra` data.
Do not place OAuth tokens or model credentials in this block.

**Step 4: Run GREEN.**

Run:

```bash
pytest tests/gateway/test_slack_mention.py -v
```

Expected: PASS.

## Task 4: Fetch minimal pre-routing thread context

**Objective:** Supply the classifier with enough speaker-aware context without fetching or exposing entire channel history.

**Files:**
- Modify: `gateway/platforms/slack.py`
- Modify: `tests/gateway/test_slack.py`

**Step 1: Write failing tests.**

Test a helper that calls Slack `conversations_replies`, excludes the current event duplicate, truncates to the configured most recent messages, resolves the bot identity, and converts messages into `ThreadMessage` records.
Test Slack API failure returns an empty sequence rather than raising.

**Step 2: Run RED.**

Run:

```bash
pytest tests/gateway/test_slack.py::TestThreadReplyHandling -v
```

Expected: FAIL because the pre-routing context helper does not exist.

**Step 3: Implement the helper.**

Reuse Slack's `conversations_replies` API but add a new structured helper rather than parsing the existing text-only `_fetch_thread_context()` result.
Limit input to the latest `context_messages` messages and cap individual message text to 1,000 characters.
Do not include attachments, tool output, or messages from outside the current thread.

**Step 4: Run GREEN.**

Run:

```bash
pytest tests/gateway/test_slack.py::TestThreadReplyHandling -v
```

Expected: PASS.

## Task 5: Insert semantic routing after deterministic gates

**Objective:** Route only ambiguous unmentioned replies from bot-participated threads through the small model.

**Files:**
- Modify: `gateway/platforms/slack.py`
- Modify: `tests/gateway/test_slack.py`
- Modify: `tests/gateway/test_slack_mention.py`

**Step 1: Write failing integration tests.**

Add cases that assert:

1. `@mention` and leading `모닥아` do not call the semantic router.
2. A new top-level unmentioned channel message does not call it.
3. `모닥이에게 시켜볼까요?` does not call it and remains silent.
4. A human-addressed message such as `카일님 의견은?` does not call it and remains silent.
5. An ambiguous eligible thread message calls it once with the eight-message context.
6. In `shadow` mode, an `ignore` semantic result is logged but does not alter the deterministic decision.
7. In `enforce` mode, `respond` at 0.85 passes through to `handle_message`.
8. In `enforce` mode, `respond` below 0.85, `ignore`, `uncertain`, timeout, and invalid JSON remain silent.

**Step 2: Run RED.**

Run:

```bash
pytest tests/gateway/test_slack.py tests/gateway/test_slack_mention.py -v
```

Expected: FAIL because routing does not yet invoke the router.

**Step 3: Implement the integration.**

Split the current `_slack_should_auto_respond_in_thread()` responsibility into:

```python
class DeterministicThreadRoute(Enum):
    RESPOND = "respond"
    IGNORE = "ignore"
    AMBIGUOUS = "ambiguous"
```

Do not send deterministic `RESPOND` or `IGNORE` cases to an LLM.
For `AMBIGUOUS`, call the semantic router only when the Slack thread is already in scope.
Use `asyncio.wait_for` with the configured timeout in addition to the auxiliary client timeout.
Log only `thread_ts`, mode, decision, confidence, and bounded reason.
Never log complete thread text at INFO level.

**Step 4: Run GREEN.**

Run:

```bash
pytest tests/gateway/test_slack.py tests/gateway/test_slack_mention.py tests/agent/test_semantic_router.py -v
```

Expected: PASS.

## Task 6: Enable shadow mode in the Modakbul profile and evaluate

**Objective:** Validate real behavior without letting the classifier alter Slack replies.

**Files:**
- Modify: `/Users/grab/.hermes/profiles/modakbul/config.yaml`
- Read: `/Users/grab/.hermes/profiles/modakbul/logs/gateway.log`

**Step 1: Add only the shadow configuration.**

```yaml
slack:
  semantic_thread_routing:
    enabled: true
    mode: shadow
    provider: openai-codex
    model: gpt-5.4-mini
    context_messages: 8
    confidence_threshold: 0.85
    timeout_seconds: 3
```

**Step 2: Restart and verify the gateway.**

Run:

```bash
hermes --profile modakbul gateway restart
hermes --profile modakbul gateway status
```

Expected: running gateway with no authentication or configuration error.

**Step 3: Evaluate labelled real cases.**

For at least 30 naturally occurring eligible thread messages, manually label the intended result before inspecting the classifier decision.
Track false positive, false negative, uncertain rate, timeout rate, and median latency.
A false positive is a reply to human-to-human conversation and is weighted more heavily than a false negative.

**Acceptance threshold for enforcement:** zero false positives in the labelled set, at least 80% recall of clearly bot-directed unmentioned follow-ups, and p95 classifier latency at or below three seconds.

## Task 7: Switch to enforce mode only after review

**Objective:** Make semantic decisions active after measured validation.

**Files:**
- Modify: `/Users/grab/.hermes/profiles/modakbul/config.yaml`

**Step 1: Change `mode: shadow` to `mode: enforce`.**

**Step 2: Restart and re-verify.**

Run:

```bash
hermes --profile modakbul gateway restart
hermes --profile modakbul gateway status
```

**Step 3: Conduct manual acceptance checks in a test thread.**

```text
모닥아, 지금부터 이 스레드에서 논의하자.        -> responds
그럼 방금 근거를 더 자세히 설명해줘.            -> responds
카일님, 이 안은 어떻게 생각하세요?             -> silent
모닥이에게 초안 시켜볼까요?                    -> silent
그럼 그걸로 가자.                               -> normally silent unless clear context earns >= 0.85
```

## Final verification

Run the focused suite:

```bash
pytest tests/agent/test_semantic_router.py tests/gateway/test_slack.py tests/gateway/test_slack_mention.py -v
```

Run the complete gateway suite:

```bash
pytest tests/gateway -q
```

Run the full test suite before committing:

```bash
python -m pytest tests/ -o 'addopts=' -q
```

Commit source code separately from profile configuration.
Use a commit message such as:

```text
feat(slack): add semantic routing for thread replies
```
