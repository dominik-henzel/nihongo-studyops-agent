"""Descriptive ADK tools with strict Pydantic v2 validation and guided error recovery.

Exposes four domain-specific tools for the Nihongo StudyOps multi-agent system:
1. ``extract_unknown_jlpt_vocabulary_from_text``
2. ``search_jlpt_grammar_vector_store``
3. ``schedule_spaced_repetition_flashcard_batch``
4. ``purge_and_reset_learner_srs_deck_critical``

Every tool validates its inputs with strict Pydantic v2 models (`extra="forbid"`),
catches ``ValidationError`` to return descriptive ``recovery_instructions`` back to
the LLM instead of raising unhandled exceptions, records OpenTelemetry spans and
``INTENT``/``OUTCOME`` JSON logs, and scrubs PII prior to storage.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pydantic import ValidationError

from .guardrails import before_tool_hitl_callback, evaluate_input_guardrails, self_evaluate_study_pack_output
from .memory import (
    get_default_memory_store,
    schedule_background_memory_consolidation,
)
from .observability import observe_tool_execution, scrub_pii, trace_span
from .schemas import (
    FlashcardBatchInput,
    FlashcardBatchOutput,
    FlashcardItem,
    GrammarPointMatch,
    GrammarSearchInput,
    GrammarSearchOutput,
    ResetLearnerDeckInput,
    ResetLearnerDeckOutput,
    VocabularyExtractionInput,
    VocabularyExtractionOutput,
    VocabularyItem,
)


def _format_validation_recovery(tool_name: str, exc: ValidationError, hint: str) -> str:
    """Format a Pydantic ``ValidationError`` into actionable LLM recovery instructions."""
    error_summaries: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = str(err.get("msg", "Invalid value"))
        error_summaries.append(f"parameter '{loc}': {msg}")
    joined_errors = "; ".join(error_summaries)
    return (
        f"Tool '{tool_name}' failed input schema validation ({joined_errors}). "
        f"RECOVERY INSTRUCTION: {hint}"
    )


@observe_tool_execution(
    tool_name="extract_unknown_jlpt_vocabulary_from_text",
    agent_name="vocabulary_miner_agent",
)
def extract_unknown_jlpt_vocabulary_from_text(
    japanese_text: str,
    target_jlpt_level: str = "N3",
    learner_id: str = "default_learner",
    max_new_words: int = 8,
    tool_context: Optional[Any] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Extract unknown i+1 Japanese vocabulary from a passage while filtering mastered words.

    Consults the learner's persistent vocabulary store to skip words already mastered,
    ranks candidate words by proximity to ``target_jlpt_level``, and schedules
    non-blocking background memory consolidation via ``asyncio.create_task()``.

    Args:
        japanese_text: Raw Japanese reading passage containing Kanji, Hiragana, or Katakana.
        target_jlpt_level: Target JLPT proficiency level ('N5', 'N4', 'N3', 'N2', or 'N1').
        learner_id: Alphanumeric learner identifier used to filter out mastered words.
        max_new_words: Maximum number of unknown vocabulary items to extract (1 to 25).
        tool_context: Optional ADK ToolContext used to store extracted vocabulary in session state.
        extra_fields: Optional dictionary of extra fields (rejected by strict Pydantic schema).

    Returns:
        Dictionary serialized from ``VocabularyExtractionOutput`` containing extracted
        ``VocabularyItem`` records, ``skipped_mastered_words``, background consolidation
        status, and ``recovery_instructions`` if input validation failed.
    """
    raw_payload: dict[str, Any] = {
        "japanese_text": japanese_text,
        "target_jlpt_level": target_jlpt_level,
        "learner_id": learner_id,
        "max_new_words": max_new_words,
    }
    if extra_fields:
        raw_payload.update(extra_fields)

    try:
        validated = VocabularyExtractionInput(**raw_payload)
    except ValidationError as exc:
        recovery = _format_validation_recovery(
            tool_name="extract_unknown_jlpt_vocabulary_from_text",
            exc=exc,
            hint=(
                "Pass a non-empty 'japanese_text' containing Japanese script (Kanji/Kana), "
                "set 'target_jlpt_level' to one of ['N5', 'N4', 'N3', 'N2', 'N1'], "
                "use an alphanumeric 'learner_id', and set 'max_new_words' between 1 and 25."
            ),
        )
        return VocabularyExtractionOutput(
            status="VALIDATION_ERROR",
            learner_id=str(learner_id or "default_learner"),
            target_jlpt_level=str(target_jlpt_level or "N3"),
            extracted_vocabulary=[],
            skipped_mastered_words=[],
            async_consolidation_scheduled=False,
            recovery_instructions=recovery,
        ).model_dump()

    store = get_default_memory_store()
    extracted, skipped_mastered = store.extract_candidate_vocabulary(
        japanese_text=validated.japanese_text,
        target_jlpt_level=validated.target_jlpt_level,
        learner_id=validated.learner_id,
        max_new_words=validated.max_new_words,
    )

    scheduled_async = schedule_background_memory_consolidation(
        learner_id=validated.learner_id,
        extracted_words=[item.surface for item in extracted],
        target_jlpt_level=validated.target_jlpt_level,
        store=store,
    )

    output = VocabularyExtractionOutput(
        status="SUCCESS",
        learner_id=validated.learner_id,
        target_jlpt_level=validated.target_jlpt_level,
        extracted_vocabulary=extracted,
        skipped_mastered_words=skipped_mastered,
        async_consolidation_scheduled=scheduled_async,
        recovery_instructions=None,
    ).model_dump()

    if tool_context is not None and hasattr(tool_context, "state"):
        try:
            tool_context.state["extracted_vocabulary"] = output["extracted_vocabulary"]
            tool_context.state["learner_id"] = validated.learner_id
            tool_context.state["target_jlpt_level"] = validated.target_jlpt_level
        except Exception:
            pass

    return output


@observe_tool_execution(
    tool_name="search_jlpt_grammar_vector_store",
    agent_name="grammar_analyzer_agent",
)
def search_jlpt_grammar_vector_store(
    japanese_text: str,
    target_jlpt_level: str = "N3",
    top_k: int = 4,
    tool_context: Optional[Any] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Query the persistent JLPT grammar vector store / corpus for matching grammar patterns.

    Searches Vertex AI Search (Discovery Engine) when configured and automatically
    falls back to the local SQLite JLPT grammar corpus when running offline.

    Args:
        japanese_text: Japanese passage or sentence to analyze for JLPT grammar constructions.
        target_jlpt_level: Target JLPT proficiency tier ('N5', 'N4', 'N3', 'N2', or 'N1').
        top_k: Maximum number of matching grammar constructions to return (1 to 10).
        tool_context: Optional ADK ToolContext used to store matched grammar points in session state.
        extra_fields: Optional dictionary of extra fields (rejected by strict Pydantic schema).

    Returns:
        Dictionary serialized from ``GrammarSearchOutput`` containing matched
        ``GrammarPointMatch`` entries, ``corpus_backend``, and ``recovery_instructions``
        on validation errors.
    """
    raw_payload: dict[str, Any] = {
        "japanese_text": japanese_text,
        "target_jlpt_level": target_jlpt_level,
        "top_k": top_k,
    }
    if extra_fields:
        raw_payload.update(extra_fields)

    try:
        validated = GrammarSearchInput(**raw_payload)
    except ValidationError as exc:
        recovery = _format_validation_recovery(
            tool_name="search_jlpt_grammar_vector_store",
            exc=exc,
            hint=(
                "Provide a 'japanese_text' string containing Japanese characters, "
                "set 'target_jlpt_level' to one of ['N5', 'N4', 'N3', 'N2', 'N1'], "
                "and keep 'top_k' between 1 and 10."
            ),
        )
        return GrammarSearchOutput(
            status="VALIDATION_ERROR",
            target_jlpt_level=str(target_jlpt_level or "N3"),
            matched_grammar_points=[],
            corpus_backend="validation_halted",
            recovery_instructions=recovery,
        ).model_dump()

    store = get_default_memory_store()
    matches, backend_used = store.search_grammar_corpus(
        japanese_text=validated.japanese_text,
        target_jlpt_level=validated.target_jlpt_level,
        top_k=validated.top_k,
    )

    output = GrammarSearchOutput(
        status="SUCCESS",
        target_jlpt_level=validated.target_jlpt_level,
        matched_grammar_points=matches,
        corpus_backend=backend_used,
        recovery_instructions=None,
    ).model_dump()

    if tool_context is not None and hasattr(tool_context, "state"):
        try:
            tool_context.state["grammar_analysis"] = output["matched_grammar_points"]
        except Exception:
            pass

    return output


@observe_tool_execution(
    tool_name="schedule_spaced_repetition_flashcard_batch",
    agent_name="srs_deck_builder_agent",
)
def schedule_spaced_repetition_flashcard_batch(
    learner_id: str = "default_learner",
    deck_name: str = "JLPT_Study_Pack",
    vocabulary_items: Optional[list[dict[str, Any]]] = None,
    grammar_points: Optional[list[dict[str, Any]]] = None,
    initial_ease_factor: float = 2.5,
    tool_context: Optional[Any] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Schedule SM-2 spaced-repetition intervals and persist Anki flashcards for a learner.

    Converts extracted vocabulary items and grammar constructions into flashcards
    with SM-2 interval parameters, formats ready-to-import Anki TSV lines
    (``Front<TAB>Back<TAB>Tags``), and persists the batch to Cloud Firestore or SQLite.

    Args:
        learner_id: Alphanumeric learner identifier owning the target flashcard deck.
        deck_name: Target spaced-repetition deck name (defaults to 'JLPT_Study_Pack').
        vocabulary_items: List of vocabulary dicts matching ``VocabularyItem`` schema.
        grammar_points: List of grammar dicts matching ``GrammarPointMatch`` schema.
        initial_ease_factor: Initial SM-2 ease factor between 1.3 and 3.5 (default 2.5).
        tool_context: Optional ADK ToolContext providing fallback state from upstream stages.
        extra_fields: Optional dictionary of extra fields (rejected by strict Pydantic schema).

    Returns:
        Dictionary serialized from ``FlashcardBatchOutput`` with scheduled flashcards,
        ``anki_tsv_export``, ``persistence_backend``, and ``recovery_instructions`` on error.
    """
    resolved_vocab = vocabulary_items
    resolved_grammar = grammar_points
    if tool_context is not None and hasattr(tool_context, "state"):
        if resolved_vocab is None:
            resolved_vocab = tool_context.state.get("extracted_vocabulary", [])
        if resolved_grammar is None:
            resolved_grammar = tool_context.state.get("grammar_analysis", [])

    raw_payload: dict[str, Any] = {
        "learner_id": learner_id,
        "deck_name": deck_name,
        "vocabulary_items": resolved_vocab or [],
        "grammar_points": resolved_grammar or [],
        "initial_ease_factor": initial_ease_factor,
    }
    if extra_fields:
        raw_payload.update(extra_fields)

    try:
        validated = FlashcardBatchInput(**raw_payload)
    except ValidationError as exc:
        recovery = _format_validation_recovery(
            tool_name="schedule_spaced_repetition_flashcard_batch",
            exc=exc,
            hint=(
                "Ensure 'learner_id' is 2-64 alphanumeric characters, 'initial_ease_factor' "
                "is between 1.3 and 3.5, and 'vocabulary_items' / 'grammar_points' follow "
                "the exact VocabularyItem and GrammarPointMatch schemas without extra fields."
            ),
        )
        return FlashcardBatchOutput(
            status="VALIDATION_ERROR",
            learner_id=str(learner_id or "default_learner"),
            deck_name=str(deck_name or "JLPT_Study_Pack"),
            scheduled_cards_count=0,
            scheduled_cards=[],
            anki_tsv_export="",
            persistence_backend="validation_halted",
            recovery_instructions=recovery,
        ).model_dump()

    if not validated.vocabulary_items and not validated.grammar_points:
        return FlashcardBatchOutput(
            status="VALIDATION_ERROR",
            learner_id=validated.learner_id,
            deck_name=validated.deck_name,
            scheduled_cards_count=0,
            scheduled_cards=[],
            anki_tsv_export="",
            persistence_backend="validation_halted",
            recovery_instructions=(
                "Tool 'schedule_spaced_repetition_flashcard_batch' received empty "
                "'vocabulary_items' and 'grammar_points'. RECOVERY INSTRUCTION: Run "
                "'extract_unknown_jlpt_vocabulary_from_text' or 'search_jlpt_grammar_vector_store' "
                "first and pass at least one vocabulary or grammar item to schedule."
            ),
        ).model_dump()

    now_utc = datetime.now(timezone.utc)
    next_review_iso = (now_utc + timedelta(days=1)).isoformat()
    scheduled_cards: list[FlashcardItem] = []

    for vocab in validated.vocabulary_items:
        digest = hashlib.sha1(f"vocab:{vocab.surface}".encode("utf-8")).hexdigest()[:10]
        card_id = f"vocab-{digest}"
        front = vocab.surface
        back = f"{vocab.reading} ({vocab.part_of_speech}) — {vocab.english_gloss}"
        tags = f"JLPT::{vocab.jlpt_level} StudyOps::Vocabulary"
        tsv_row = f"{front}\t{back}\t{tags}"
        scheduled_cards.append(
            FlashcardItem(
                card_id=card_id,
                front_japanese=front,
                back_reading_and_meaning=back,
                example_bilingual=vocab.context_sentence,
                jlpt_level=vocab.jlpt_level,
                ease_factor=validated.initial_ease_factor,
                interval_days=1,
                next_review_iso=next_review_iso,
                anki_tsv_row=tsv_row,
            )
        )

    for gram in validated.grammar_points:
        digest = hashlib.sha1(f"grammar:{gram.grammar_id}".encode("utf-8")).hexdigest()[:10]
        card_id = f"grammar-{digest}"
        front = f"{gram.pattern} [{gram.jlpt_level}]"
        back = f"{gram.english_meaning} | Rule: {gram.formation_rule}"
        bilingual = f"{gram.japanese_example} ({gram.english_translation})"
        tags = f"JLPT::{gram.jlpt_level} StudyOps::Grammar"
        tsv_row = f"{front}\t{back} — {bilingual}\t{tags}"
        scheduled_cards.append(
            FlashcardItem(
                card_id=card_id,
                front_japanese=front,
                back_reading_and_meaning=back,
                example_bilingual=bilingual,
                jlpt_level=gram.jlpt_level,
                ease_factor=validated.initial_ease_factor,
                interval_days=1,
                next_review_iso=next_review_iso,
                anki_tsv_row=tsv_row,
            )
        )

    store = get_default_memory_store()
    backend_used = store.save_flashcards(
        learner_id=validated.learner_id,
        deck_name=validated.deck_name,
        cards=scheduled_cards,
    )

    anki_tsv_export = "\n".join(card.anki_tsv_row for card in scheduled_cards)
    output = FlashcardBatchOutput(
        status="SUCCESS",
        learner_id=validated.learner_id,
        deck_name=validated.deck_name,
        scheduled_cards_count=len(scheduled_cards),
        scheduled_cards=scheduled_cards,
        anki_tsv_export=anki_tsv_export,
        persistence_backend=backend_used,
        recovery_instructions=None,
    ).model_dump()

    if tool_context is not None and hasattr(tool_context, "state"):
        try:
            tool_context.state["scheduled_deck_summary"] = output
        except Exception:
            pass

    return output


@observe_tool_execution(
    tool_name="purge_and_reset_learner_srs_deck_critical",
    agent_name="curriculum_planner_agent",
)
def purge_and_reset_learner_srs_deck_critical(
    learner_id: str,
    deck_name: str = "JLPT_Study_Pack",
    reset_reason: str = "",
    human_confirmation_approved: bool = False,
    tool_context: Optional[Any] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Permanently purge and reset a learner's spaced-repetition deck after HITL confirmation.

    High-stakes destructive action gated by ``before_tool_hitl_callback``. Refuses to
    delete learner progress unless ``human_confirmation_approved=True`` or
    ``tool_context.state['hitl_approved']`` is explicitly set.

    Args:
        learner_id: Alphanumeric learner identifier whose SRS deck will be purged.
        deck_name: Target deck name to reset (defaults to 'JLPT_Study_Pack').
        reset_reason: Explicit pedagogical reason (at least 5 characters) for resetting.
        human_confirmation_approved: Boolean flag that must be True after human approval.
        tool_context: Optional ADK ToolContext carrying session-level HITL approval state.
        extra_fields: Optional dictionary of extra fields (rejected by strict Pydantic schema).

    Returns:
        Dictionary serialized from ``ResetLearnerDeckOutput`` reporting either
        ``status="PENDING_HUMAN_APPROVAL"``, ``status="VALIDATION_ERROR"``, or
        ``status="SUCCESS"`` with counts of purged records.
    """
    raw_payload: dict[str, Any] = {
        "learner_id": learner_id,
        "deck_name": deck_name,
        "reset_reason": reset_reason,
        "human_confirmation_approved": human_confirmation_approved,
    }
    if extra_fields:
        raw_payload.update(extra_fields)

    try:
        validated = ResetLearnerDeckInput(**raw_payload)
    except ValidationError as exc:
        recovery = _format_validation_recovery(
            tool_name="purge_and_reset_learner_srs_deck_critical",
            exc=exc,
            hint=(
                "Provide a valid alphanumeric 'learner_id', a non-empty 'deck_name', "
                "a descriptive 'reset_reason' (at least 5 characters), and boolean "
                "'human_confirmation_approved'."
            ),
        )
        return ResetLearnerDeckOutput(
            status="VALIDATION_ERROR",
            learner_id=str(learner_id or "default_learner"),
            deck_name=str(deck_name or "JLPT_Study_Pack"),
            purged_cards_count=0,
            purged_vocabulary_count=0,
            human_approved=False,
            recovery_instructions=recovery,
        ).model_dump()

    hitl_intercept = before_tool_hitl_callback(
        tool="purge_and_reset_learner_srs_deck_critical",
        args=validated.model_dump(),
        tool_context=tool_context,
    )
    if hitl_intercept is not None:
        return hitl_intercept

    store = get_default_memory_store()
    purged_cards, purged_vocab = store.purge_learner_deck(
        learner_id=validated.learner_id,
        deck_name=validated.deck_name,
    )

    return ResetLearnerDeckOutput(
        status="SUCCESS",
        learner_id=validated.learner_id,
        deck_name=validated.deck_name,
        purged_cards_count=purged_cards,
        purged_vocabulary_count=purged_vocab,
        human_approved=True,
        recovery_instructions=None,
    ).model_dump()


def run_study_pack_pipeline_offline(
    japanese_text: str,
    target_jlpt_level: str = "N3",
    learner_id: str = "default_learner",
    deck_name: str = "JLPT_Study_Pack",
) -> dict[str, Any]:
    """Execute the 3-stage study-pack sequential pipeline deterministically offline.

    Runs input guardrails, mines unknown i+1 vocabulary, retrieves matching JLPT
    grammar patterns, schedules SM-2 Anki flashcards, and performs post-generation
    self-evaluation inside a parent OpenTelemetry trace span.
    """
    with trace_span(
        "pipeline.study_pack_sequential",
        attributes={"learner.id": learner_id, "jlpt.level": target_jlpt_level},
    ) as root_span:
        input_check = evaluate_input_guardrails(japanese_text)
        if not input_check.allowed:
            return {
                "status": "BLOCKED_BY_GUARDRAIL",
                "reason": input_check.reason,
                "triggered_rule": input_check.triggered_rule,
                "trace_id": root_span["trace_id"],
            }

        vocab_out = extract_unknown_jlpt_vocabulary_from_text(
            japanese_text=input_check.sanitized_text,
            target_jlpt_level=target_jlpt_level,
            learner_id=learner_id,
        )
        if vocab_out["status"] != "SUCCESS":
            return {
                "status": vocab_out["status"],
                "stage": "vocabulary_miner_agent",
                "recovery_instructions": vocab_out.get("recovery_instructions"),
                "trace_id": root_span["trace_id"],
            }

        grammar_out = search_jlpt_grammar_vector_store(
            japanese_text=input_check.sanitized_text,
            target_jlpt_level=target_jlpt_level,
        )
        if grammar_out["status"] != "SUCCESS":
            return {
                "status": grammar_out["status"],
                "stage": "grammar_analyzer_agent",
                "recovery_instructions": grammar_out.get("recovery_instructions"),
                "trace_id": root_span["trace_id"],
            }

        deck_out = schedule_spaced_repetition_flashcard_batch(
            learner_id=learner_id,
            deck_name=deck_name,
            vocabulary_items=vocab_out["extracted_vocabulary"],
            grammar_points=grammar_out["matched_grammar_points"],
        )

        summary_markdown = (
            f"### JLPT {target_jlpt_level} Study Pack ({learner_id})\n"
            f"- Passage: {scrub_pii(input_check.sanitized_text)}\n"
            f"- Extracted Vocabulary: {len(vocab_out['extracted_vocabulary'])}\n"
            f"- Matched Grammar Patterns: {len(grammar_out['matched_grammar_points'])}\n"
            f"- Scheduled Flashcards: {deck_out['scheduled_cards_count']}\n"
        )
        self_eval = self_evaluate_study_pack_output(summary_markdown)

        return {
            "status": deck_out["status"],
            "learner_id": learner_id,
            "target_jlpt_level": target_jlpt_level,
            "vocabulary": vocab_out,
            "grammar": grammar_out,
            "flashcards": deck_out,
            "self_evaluation": {
                "allowed": self_eval.allowed,
                "status": self_eval.status,
                "summary_markdown": self_eval.sanitized_text,
            },
            "trace_id": root_span["trace_id"],
        }
