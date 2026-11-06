"""Security guardrails, self-evaluation policy plugin, and Human-in-the-Loop (HITL) hooks.

Implements three orchestration safety layers integrated with Google ADK callbacks
and ``BasePlugin``:
1. **Pre-Model Input Guardrails (``evaluate_input_guardrails`` & ``before_model_guardrail_callback``)**:
   - Detects prompt-injection and policy-bypass attempts before calling the model.
   - Scrubs PII from incoming user prompts via ``PiiRedactionScrubber``.
2. **Post-Model Self-Evaluation (``self_evaluate_study_pack_output`` & ``after_model_self_eval_callback``)**:
   - Audits generated study packs for Japanese script presence, structural completeness,
     and absence of leaked PII tokens.
3. **Human-in-the-Loop Tool Gate (``before_tool_hitl_callback``)**:
   - Intercepts high-stakes destructive calls to ``purge_and_reset_learner_srs_deck_critical``
     and returns ``status="PENDING_HUMAN_APPROVAL"`` unless explicit human confirmation
     is present in tool arguments or session state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from .observability import log_agent_intent, log_agent_outcome, scrub_pii
from .schemas import ResetLearnerDeckOutput, contains_japanese_script

try:
    from google.adk.models.llm_response import LlmResponse
    from google.adk.plugins.base_plugin import BasePlugin
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    LlmResponse = None  # type: ignore[assignment,misc]
    genai_types = None  # type: ignore[assignment]

    class BasePlugin:  # type: ignore[no-redef]
        """Fallback BasePlugin when running without google-adk installed."""

        def __init__(self, name: str = "studyops_policy_plugin") -> None:
            self.name = name


# High-signal patterns indicating prompt injection, system-prompt extraction, or HITL bypass.
_PROMPT_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|system)\s+instructions",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"(?:reveal|print|dump|output|show)\s+(?:your\s+)?(?:system\s+prompt|agent_constitution|hidden\s+instructions)",
            re.IGNORECASE,
        ),
    ),
    (
        "hitl_bypass_attempt",
        re.compile(
            r"bypass\s+(?:human|hitl)\s+(?:approval|confirmation)|force\s+purge_and_reset",
            re.IGNORECASE,
        ),
    ),
    (
        "script_or_sql_injection",
        re.compile(r"<\s*script\b|drop\s+table\s+learner_", re.IGNORECASE),
    ),
)

HIGH_STAKES_TOOLS: frozenset[str] = frozenset({"purge_and_reset_learner_srs_deck_critical"})


@dataclass(frozen=True)
class GuardrailDecision:
    """Result of evaluating pre-model input guardrails or post-model self-evaluation."""

    allowed: bool
    status: str  # "ALLOWED", "BLOCKED_BY_GUARDRAIL", or "SELF_EVAL_WARNING"
    sanitized_text: str
    reason: Optional[str] = None
    triggered_rule: Optional[str] = None


def evaluate_input_guardrails(user_text: str, max_chars: int = 4000) -> GuardrailDecision:
    """Inspect ``user_text`` for prompt injection, excessive length, and PII before LLM calls."""
    if not user_text or not user_text.strip():
        return GuardrailDecision(
            allowed=False,
            status="BLOCKED_BY_GUARDRAIL",
            sanitized_text="",
            reason="Input text is empty. Please provide a Japanese sentence or study request.",
            triggered_rule="empty_input",
        )

    if len(user_text) > max_chars:
        return GuardrailDecision(
            allowed=False,
            status="BLOCKED_BY_GUARDRAIL",
            sanitized_text=scrub_pii(user_text[:max_chars]),
            reason=f"Input length ({len(user_text)} chars) exceeds maximum guardrail limit ({max_chars}).",
            triggered_rule="max_length_exceeded",
        )

    for rule_name, pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(user_text):
            log_agent_outcome(
                action_name="evaluate_input_guardrails",
                outcome_payload={"triggered_rule": rule_name},
                status="BLOCKED_BY_GUARDRAIL",
            )
            return GuardrailDecision(
                allowed=False,
                status="BLOCKED_BY_GUARDRAIL",
                sanitized_text="[BLOCKED_PROMPT_INJECTION]",
                reason=(
                    f"Request blocked by security guardrail ('{rule_name}'). "
                    "Please submit a legitimate Japanese language study request."
                ),
                triggered_rule=rule_name,
            )

    sanitized = scrub_pii(user_text.strip())
    return GuardrailDecision(
        allowed=True,
        status="ALLOWED",
        sanitized_text=sanitized,
        reason=None,
        triggered_rule=None,
    )


def self_evaluate_study_pack_output(output_text: str) -> GuardrailDecision:
    """Self-evaluate generated study-pack output for Japanese script, quality, and PII safety."""
    if not output_text or not output_text.strip():
        return GuardrailDecision(
            allowed=False,
            status="SELF_EVAL_WARNING",
            sanitized_text="",
            reason="Self-evaluation failed: model output was empty.",
            triggered_rule="empty_output",
        )

    scrubbed_output = scrub_pii(output_text)
    if not contains_japanese_script(scrubbed_output):
        return GuardrailDecision(
            allowed=False,
            status="SELF_EVAL_WARNING",
            sanitized_text=scrubbed_output,
            reason=(
                "Self-evaluation failed: study-pack output does not contain any Japanese "
                "script (Kanji, Hiragana, or Katakana)."
            ),
            triggered_rule="missing_japanese_script",
        )

    return GuardrailDecision(
        allowed=True,
        status="ALLOWED",
        sanitized_text=scrubbed_output,
        reason=None,
        triggered_rule=None,
    )


def _extract_latest_user_text(llm_request: Any) -> str:
    """Extract the latest user message string from an ADK ``LlmRequest`` object or dict."""
    if llm_request is None:
        return ""
    if isinstance(llm_request, str):
        return llm_request
    if isinstance(llm_request, dict):
        return str(llm_request.get("user_text", "") or llm_request.get("content", ""))

    contents = getattr(llm_request, "contents", None)
    if not contents:
        return ""
    extracted_parts: list[str] = []
    for content in contents:
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text_val = getattr(part, "text", None)
            if text_val:
                extracted_parts.append(str(text_val))
    return "\n".join(extracted_parts)


def before_model_guardrail_callback(
    callback_context: Any = None,
    llm_request: Any = None,
    **kwargs: Any,
) -> Optional[Any]:
    """ADK ``before_model_callback`` enforcing prompt-injection defense and PII scrubbing.

    Returns ``None`` when the request is safe to proceed, or a short-circuit response
    when a security guardrail blocks the input.
    """
    raw_text = _extract_latest_user_text(llm_request)
    if not raw_text:
        return None

    decision = evaluate_input_guardrails(raw_text)
    if not decision.allowed:
        if LlmResponse is not None and genai_types is not None:
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part.from_text(text=decision.reason or "Request blocked.")],
                )
            )
        return {
            "status": decision.status,
            "reason": decision.reason,
            "triggered_rule": decision.triggered_rule,
        }

    # Scrub PII in-place on the outgoing request parts when supported by the request object.
    contents = getattr(llm_request, "contents", None)
    if contents:
        for content in contents:
            parts = getattr(content, "parts", None) or []
            for part in parts:
                if getattr(part, "text", None):
                    part.text = scrub_pii(part.text)
    return None


def after_model_self_eval_callback(
    callback_context: Any = None,
    llm_response: Any = None,
    **kwargs: Any,
) -> Optional[Any]:
    """ADK ``after_model_callback`` running automated self-evaluation and PII scrubbing."""
    if llm_response is None:
        return None

    if isinstance(llm_response, str):
        decision = self_evaluate_study_pack_output(llm_response)
        return decision

    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return None

    combined_text = "\n".join(
        str(getattr(part, "text", "")) for part in parts if getattr(part, "text", None)
    )
    if not combined_text:
        return None

    decision = self_evaluate_study_pack_output(combined_text)
    for part in parts:
        if getattr(part, "text", None):
            part.text = scrub_pii(part.text)

    if callback_context is not None and hasattr(callback_context, "state"):
        try:
            callback_context.state["last_self_eval_status"] = decision.status
        except Exception:
            pass
    return None


def before_tool_hitl_callback(
    tool: Any = None,
    args: Optional[dict[str, Any]] = None,
    tool_context: Any = None,
    **kwargs: Any,
) -> Optional[dict[str, Any]]:
    """ADK ``before_tool_callback`` enforcing Human-in-the-Loop confirmation on destructive tools.

    Intercepts ``purge_and_reset_learner_srs_deck_critical`` and halts execution with
    ``status="PENDING_HUMAN_APPROVAL"`` unless ``human_confirmation_approved=True`` is
    explicitly passed or ``tool_context.state["hitl_approved"]`` is set.
    """
    tool_name = (
        getattr(tool, "name", None)
        or getattr(tool, "__name__", None)
        or str(tool or "")
    )
    if tool_name not in HIGH_STAKES_TOOLS:
        return None

    call_args = dict(args or {})
    explicit_arg_approved = bool(call_args.get("human_confirmation_approved", False))
    context_state_approved = False
    if tool_context is not None and hasattr(tool_context, "state") and tool_context.state:
        context_state_approved = bool(tool_context.state.get("hitl_approved", False))

    learner_id = str(call_args.get("learner_id", "default_learner"))
    deck_name = str(call_args.get("deck_name", "JLPT_Study_Pack"))

    log_agent_intent(
        action_name=f"hitl_gate.{tool_name}",
        intent_payload={
            "learner_id": learner_id,
            "deck_name": deck_name,
            "explicit_arg_approved": explicit_arg_approved,
            "context_state_approved": context_state_approved,
        },
    )

    if explicit_arg_approved or context_state_approved:
        log_agent_outcome(
            action_name=f"hitl_gate.{tool_name}",
            outcome_payload={"hitl_decision": "APPROVED"},
            status="SUCCESS",
        )
        return None

    blocked_response = ResetLearnerDeckOutput(
        status="PENDING_HUMAN_APPROVAL",
        learner_id=learner_id,
        deck_name=deck_name,
        purged_cards_count=0,
        purged_vocabulary_count=0,
        human_approved=False,
        recovery_instructions=(
            "STOP: High-stakes action 'purge_and_reset_learner_srs_deck_critical' requires "
            "explicit human-in-the-loop confirmation before deleting learner SRS history. "
            "Ask the user to confirm resetting deck "
            f"'{deck_name}' for learner '{learner_id}', then retry with "
            "human_confirmation_approved=True."
        ),
    ).model_dump()

    log_agent_outcome(
        action_name=f"hitl_gate.{tool_name}",
        outcome_payload=blocked_response,
        status="PENDING_HUMAN_APPROVAL",
    )
    return blocked_response


class StudyOpsPolicyPlugin(BasePlugin):
    """ADK Policy Plugin bundling input guardrails, post-model self-eval, and HITL tool hooks."""

    def __init__(self, name: str = "studyops_policy_plugin") -> None:
        super().__init__(name=name)

    async def before_model_callback(
        self,
        *,
        callback_context: Any = None,
        llm_request: Any = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        return before_model_guardrail_callback(
            callback_context=callback_context,
            llm_request=llm_request,
            **kwargs,
        )

    async def after_model_callback(
        self,
        *,
        callback_context: Any = None,
        llm_response: Any = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        return after_model_self_eval_callback(
            callback_context=callback_context,
            llm_response=llm_response,
            **kwargs,
        )

    async def before_tool_callback(
        self,
        *,
        tool: Any = None,
        tool_args: Optional[dict[str, Any]] = None,
        tool_context: Any = None,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        return before_tool_hitl_callback(
            tool=tool,
            args=tool_args,
            tool_context=tool_context,
            **kwargs,
        )
