"""Strict Pydantic v2 input and output schemas for tool and agent contracts.

Every tool in ``nihongo_agent`` validates incoming arguments and structures outgoing
responses through these models. Using ``ConfigDict(extra="forbid")`` guarantees
that unexpected or hallucinated fields from an LLM tool call are rejected
immediately with structured validation metadata that the tool layer translates
into actionable ``recovery_instructions``.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

JLPTLevel = Literal["N5", "N4", "N3", "N2", "N1"]
ToolExecutionStatus = Literal[
    "SUCCESS",
    "VALIDATION_ERROR",
    "PENDING_HUMAN_APPROVAL",
    "BLOCKED_BY_GUARDRAIL",
    "ERROR",
]

# Matches Hiragana (\u3040-\u309F), Katakana (\u30A0-\u30FF), or CJK Kanji (\u4E00-\u9FAF).
JAPANESE_SCRIPT_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]")
LEARNER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{2,64}$")


def contains_japanese_script(text: str) -> bool:
    """Return True if ``text`` contains at least one Hiragana, Katakana, or Kanji character."""
    return bool(JAPANESE_SCRIPT_PATTERN.search(text or ""))


class VocabularyItem(BaseModel):
    """A single Japanese vocabulary entry extracted for a learner's study pack."""

    model_config = ConfigDict(extra="forbid")

    surface: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Surface form in Kanji or Kana as it appears in the passage (e.g., '再生可能').",
    )
    reading: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Hiragana reading (furigana) for pronunciation (e.g., 'さいせいかのう').",
    )
    jlpt_level: JLPTLevel = Field(
        ...,
        description="Estimated JLPT proficiency tier from N5 (beginner) to N1 (advanced).",
    )
    part_of_speech: str = Field(
        default="noun",
        min_length=1,
        max_length=64,
        description="Grammatical category such as 'noun', 'verb (ichidan)', 'adjective (i)', or 'expression'.",
    )
    english_gloss: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Concise English definition suited for an Anki flashcard back.",
    )
    context_sentence: str = Field(
        default="",
        max_length=512,
        description="Original Japanese sentence or clause where the word appeared.",
    )


class VocabularyExtractionInput(BaseModel):
    """Validated input contract for ``extract_unknown_jlpt_vocabulary_from_text``."""

    model_config = ConfigDict(extra="forbid")

    japanese_text: str = Field(
        ...,
        min_length=2,
        max_length=4000,
        description="Raw Japanese passage (containing Kanji/Kana) to mine for unknown i+1 vocabulary.",
    )
    target_jlpt_level: JLPTLevel = Field(
        default="N3",
        description="Learner's target JLPT level ('N5', 'N4', 'N3', 'N2', or 'N1').",
    )
    learner_id: str = Field(
        default="default_learner",
        min_length=2,
        max_length=64,
        description="Alphanumeric learner identifier used to filter out already-mastered words.",
    )
    max_new_words: int = Field(
        default=8,
        ge=1,
        le=25,
        description="Maximum number of unknown vocabulary items to extract in a single study pack.",
    )

    @field_validator("japanese_text")
    @classmethod
    def validate_japanese_script_present(cls, value: str) -> str:
        cleaned = value.strip()
        if not contains_japanese_script(cleaned):
            raise ValueError(
                "Input 'japanese_text' must contain Japanese script (Hiragana, Katakana, or Kanji)."
            )
        return cleaned

    @field_validator("learner_id")
    @classmethod
    def validate_learner_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not LEARNER_ID_PATTERN.match(cleaned):
            raise ValueError(
                "Input 'learner_id' must be 2-64 characters using only letters, digits, '_', or '-'."
            )
        return cleaned


class VocabularyExtractionOutput(BaseModel):
    """Structured output contract returned by ``extract_unknown_jlpt_vocabulary_from_text``."""

    model_config = ConfigDict(extra="forbid")

    status: ToolExecutionStatus = Field(
        ...,
        description="Execution status ('SUCCESS' or 'VALIDATION_ERROR').",
    )
    learner_id: str = Field(
        ...,
        description="Learner identifier whose mastered lexicon was consulted.",
    )
    target_jlpt_level: str = Field(
        ...,
        description="Target JLPT proficiency level used for vocabulary filtering.",
    )
    extracted_vocabulary: list[VocabularyItem] = Field(
        default_factory=list,
        description="Unknown i+1 vocabulary items selected for the study pack.",
    )
    skipped_mastered_words: list[str] = Field(
        default_factory=list,
        description="Surface forms skipped because the learner has already mastered them.",
    )
    async_consolidation_scheduled: bool = Field(
        default=False,
        description="True when non-blocking background memory consolidation was scheduled via asyncio.",
    )
    recovery_instructions: Optional[str] = Field(
        default=None,
        description="Actionable guidance for the calling LLM if validation or extraction failed.",
    )


class GrammarPointMatch(BaseModel):
    """A verified JLPT grammar pattern retrieved from the grammar corpus / vector store."""

    model_config = ConfigDict(extra="forbid")

    grammar_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Canonical identifier in the JLPT grammar knowledge base (e.g., 'jlpt-n3-ni-tsurete').",
    )
    pattern: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Japanese grammar structure pattern (e.g., '〜につれて').",
    )
    jlpt_level: JLPTLevel = Field(
        ...,
        description="JLPT level classification of the grammar pattern.",
    )
    formation_rule: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Morphological connection rule (e.g., 'Verb (Dictionary form) / Noun + につれて').",
    )
    english_meaning: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="English explanation of nuance and usage.",
    )
    japanese_example: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Model Japanese example sentence illustrating the pattern.",
    )
    english_translation: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="English translation of the model example sentence.",
    )
    relevance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Normalized retrieval confidence score between 0.0 and 1.0.",
    )


class GrammarSearchInput(BaseModel):
    """Validated input contract for ``search_jlpt_grammar_vector_store``."""

    model_config = ConfigDict(extra="forbid")

    japanese_text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Japanese passage or grammar query to match against the JLPT grammar corpus.",
    )
    target_jlpt_level: JLPTLevel = Field(
        default="N3",
        description="Target JLPT tier ('N5', 'N4', 'N3', 'N2', or 'N1') to prioritize in retrieval.",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of grammar patterns to retrieve (1 to 10).",
    )

    @field_validator("japanese_text")
    @classmethod
    def validate_query_script(cls, value: str) -> str:
        cleaned = value.strip()
        if not contains_japanese_script(cleaned):
            raise ValueError(
                "Input 'japanese_text' must include Japanese characters to match JLPT grammar constructions."
            )
        return cleaned


class GrammarSearchOutput(BaseModel):
    """Structured output contract returned by ``search_jlpt_grammar_vector_store``."""

    model_config = ConfigDict(extra="forbid")

    status: ToolExecutionStatus = Field(
        ...,
        description="Execution status ('SUCCESS' or 'VALIDATION_ERROR').",
    )
    target_jlpt_level: str = Field(
        ...,
        description="Target JLPT tier used during retrieval.",
    )
    matched_grammar_points: list[GrammarPointMatch] = Field(
        default_factory=list,
        description="Ordered list of matching JLPT grammar constructions with formation rules.",
    )
    corpus_backend: str = Field(
        default="local_sqlite_vector_fallback",
        description="Identifier of the search backend used ('vertex_ai_search' or 'local_sqlite_vector_fallback').",
    )
    recovery_instructions: Optional[str] = Field(
        default=None,
        description="Actionable guidance for the calling LLM when inputs fail validation.",
    )


class FlashcardItem(BaseModel):
    """A spaced-repetition flashcard scheduled with SM-2 interval parameters and Anki TSV output."""

    model_config = ConfigDict(extra="forbid")

    card_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Deterministic card identifier within the learner's deck.",
    )
    front_japanese: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Front prompt of the Anki card (Japanese expression or cloze sentence).",
    )
    back_reading_and_meaning: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Back answer containing Hiragana reading, English gloss, and formation notes.",
    )
    example_bilingual: str = Field(
        default="",
        max_length=512,
        description="Supporting bilingual example sentence.",
    )
    jlpt_level: JLPTLevel = Field(
        default="N3",
        description="JLPT tag attached to the flashcard.",
    )
    ease_factor: float = Field(
        default=2.5,
        ge=1.3,
        le=3.5,
        description="SM-2 spaced-repetition ease factor (minimum 1.3, default 2.5).",
    )
    interval_days: int = Field(
        default=1,
        ge=1,
        le=3650,
        description="Scheduled review interval in days.",
    )
    next_review_iso: str = Field(
        ...,
        min_length=10,
        max_length=64,
        description="ISO-8601 UTC timestamp for the next scheduled review.",
    )
    anki_tsv_row: str = Field(
        ...,
        min_length=3,
        max_length=1024,
        description="Tab-separated row ('Front<TAB>Back<TAB>Tags') ready for direct Anki import.",
    )


class FlashcardBatchInput(BaseModel):
    """Validated input contract for ``schedule_spaced_repetition_flashcard_batch``."""

    model_config = ConfigDict(extra="forbid")

    learner_id: str = Field(
        ...,
        min_length=2,
        max_length=64,
        description="Learner identifier owning the target spaced-repetition deck.",
    )
    deck_name: str = Field(
        default="JLPT_Study_Pack",
        min_length=1,
        max_length=128,
        description="Target Anki / SRS deck name.",
    )
    vocabulary_items: list[VocabularyItem] = Field(
        default_factory=list,
        description="Extracted vocabulary entries to convert into flashcards.",
    )
    grammar_points: list[GrammarPointMatch] = Field(
        default_factory=list,
        description="Matched grammar constructions to convert into flashcards.",
    )
    initial_ease_factor: float = Field(
        default=2.5,
        ge=1.3,
        le=3.5,
        description="Initial SM-2 ease factor between 1.3 and 3.5.",
    )

    @field_validator("learner_id")
    @classmethod
    def validate_learner_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not LEARNER_ID_PATTERN.match(cleaned):
            raise ValueError(
                "Input 'learner_id' must be 2-64 characters using only letters, digits, '_', or '-'."
            )
        return cleaned


class FlashcardBatchOutput(BaseModel):
    """Structured output contract returned by ``schedule_spaced_repetition_flashcard_batch``."""

    model_config = ConfigDict(extra="forbid")

    status: ToolExecutionStatus = Field(
        ...,
        description="Execution status ('SUCCESS' or 'VALIDATION_ERROR').",
    )
    learner_id: str = Field(
        ...,
        description="Learner identifier whose SRS deck was updated.",
    )
    deck_name: str = Field(
        ...,
        description="Name of the spaced-repetition deck.",
    )
    scheduled_cards_count: int = Field(
        default=0,
        ge=0,
        description="Total number of flashcards scheduled and persisted.",
    )
    scheduled_cards: list[FlashcardItem] = Field(
        default_factory=list,
        description="Persisted flashcards with SM-2 scheduling metadata.",
    )
    anki_tsv_export: str = Field(
        default="",
        description="Newline-delimited Anki TSV payload ('Front\\tBack\\tTags') for one-step import.",
    )
    persistence_backend: str = Field(
        default="sqlite_fallback",
        description="Storage engine used ('cloud_firestore' or 'sqlite_fallback').",
    )
    recovery_instructions: Optional[str] = Field(
        default=None,
        description="Actionable guidance for the calling LLM if scheduling failed validation.",
    )


class ResetLearnerDeckInput(BaseModel):
    """Validated input contract for ``purge_and_reset_learner_srs_deck_critical``."""

    model_config = ConfigDict(extra="forbid")

    learner_id: str = Field(
        ...,
        min_length=2,
        max_length=64,
        description="Learner identifier whose spaced-repetition deck will be permanently reset.",
    )
    deck_name: str = Field(
        default="JLPT_Study_Pack",
        min_length=1,
        max_length=128,
        description="Name of the deck to purge.",
    )
    reset_reason: str = Field(
        ...,
        min_length=5,
        max_length=512,
        description="Explicit pedagogical or administrative reason for resetting the learner's deck.",
    )
    human_confirmation_approved: bool = Field(
        default=False,
        description="Must be explicitly True after human-in-the-loop confirmation before deletion proceeds.",
    )

    @field_validator("learner_id")
    @classmethod
    def validate_learner_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not LEARNER_ID_PATTERN.match(cleaned):
            raise ValueError(
                "Input 'learner_id' must be 2-64 characters using only letters, digits, '_', or '-'."
            )
        return cleaned


class ResetLearnerDeckOutput(BaseModel):
    """Structured output contract returned by ``purge_and_reset_learner_srs_deck_critical``."""

    model_config = ConfigDict(extra="forbid")

    status: ToolExecutionStatus = Field(
        ...,
        description="Execution status ('SUCCESS', 'PENDING_HUMAN_APPROVAL', or 'VALIDATION_ERROR').",
    )
    learner_id: str = Field(
        ...,
        description="Learner identifier targeted by the reset operation.",
    )
    deck_name: str = Field(
        ...,
        description="Name of the deck targeted by the reset operation.",
    )
    purged_cards_count: int = Field(
        default=0,
        ge=0,
        description="Number of flashcards deleted when approved.",
    )
    purged_vocabulary_count: int = Field(
        default=0,
        ge=0,
        description="Number of mastered vocabulary records cleared when approved.",
    )
    human_approved: bool = Field(
        default=False,
        description="Indicates whether explicit human confirmation was verified prior to execution.",
    )
    recovery_instructions: Optional[str] = Field(
        default=None,
        description="Actionable instructions for obtaining human confirmation or fixing invalid arguments.",
    )
