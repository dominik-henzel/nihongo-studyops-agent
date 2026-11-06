"""Context compaction, Vertex AI Context Caching, persistent session state, and async memory.

Implements three memory engineering capabilities with offline SQLite fallbacks:
1. **History Compaction & Context Caching**:
   - ``compact_conversation_history()`` applies sliding-window retention, older-turn
     summarization, and strict token-budget truncation to prevent context bloat.
   - ``configure_vertex_context_cache()`` provisions a Vertex AI Context Cache
     (``google.genai``) for the static JLPT N5-N1 reference corpus.
2. **Persistent Session State & Vector/Corpus Retrieval**:
   - ``PersistentLearnerMemoryStore`` connects to Google Cloud Firestore
     (``google.cloud.firestore``) for learner SRS state and Vertex AI Search
     (``google.cloud.discoveryengine_v1``) for grammar retrieval, falling back
     automatically to a local SQLite database seeded with curated JLPT N5-N1
     vocabulary and grammar patterns.
3. **Async Memory Operations**:
   - ``consolidate_and_persist_memory_async()`` and
     ``schedule_background_memory_consolidation()`` schedule expensive memory
     consolidation on the background event loop via ``asyncio.create_task()``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from .observability import scrub_pii, trace_span
from .schemas import FlashcardItem, GrammarPointMatch, JLPTLevel, VocabularyItem

try:
    from google.cloud import firestore  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    firestore = None  # type: ignore[assignment]

try:
    from google.cloud import discoveryengine_v1  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    discoveryengine_v1 = None  # type: ignore[assignment]

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

# Ordered JLPT difficulty rank (1 = easiest N5, 5 = hardest N1).
JLPT_DIFFICULTY_RANK: dict[str, int] = {
    "N5": 1,
    "N4": 2,
    "N3": 3,
    "N2": 4,
    "N1": 5,
}

# Curated JLPT N5-N1 vocabulary seed lexicon for deterministic offline extraction.
SEED_JLPT_VOCABULARY: tuple[dict[str, str], ...] = (
    # N5 foundational vocabulary
    {
        "surface": "学生",
        "reading": "がくせい",
        "jlpt_level": "N5",
        "part_of_speech": "noun",
        "english_gloss": "student",
    },
    {
        "surface": "毎日",
        "reading": "まいにち",
        "jlpt_level": "N5",
        "part_of_speech": "adverb / noun",
        "english_gloss": "every day",
    },
    {
        "surface": "図書館",
        "reading": "としょかん",
        "jlpt_level": "N5",
        "part_of_speech": "noun",
        "english_gloss": "library",
    },
    {
        "surface": "勉強",
        "reading": "べんきょう",
        "jlpt_level": "N5",
        "part_of_speech": "noun / suru-verb",
        "english_gloss": "study; diligence",
    },
    {
        "surface": "電車",
        "reading": "でんしゃ",
        "jlpt_level": "N5",
        "part_of_speech": "noun",
        "english_gloss": "electric train",
    },
    # N4 elementary vocabulary
    {
        "surface": "習慣",
        "reading": "しゅうかん",
        "jlpt_level": "N4",
        "part_of_speech": "noun",
        "english_gloss": "habit; custom",
    },
    {
        "surface": "準備",
        "reading": "じゅんび",
        "jlpt_level": "N4",
        "part_of_speech": "noun / suru-verb",
        "english_gloss": "preparation; arrangement",
    },
    {
        "surface": "経験",
        "reading": "けいけん",
        "jlpt_level": "N4",
        "part_of_speech": "noun / suru-verb",
        "english_gloss": "experience",
    },
    {
        "surface": "増える",
        "reading": "ふえる",
        "jlpt_level": "N4",
        "part_of_speech": "verb (ichidan)",
        "english_gloss": "to increase; to multiply",
    },
    # N3 intermediate vocabulary
    {
        "surface": "再生可能",
        "reading": "さいせいかのう",
        "jlpt_level": "N3",
        "part_of_speech": "na-adjective / noun",
        "english_gloss": "renewable; regenerable",
    },
    {
        "surface": "環境",
        "reading": "かんきょう",
        "jlpt_level": "N3",
        "part_of_speech": "noun",
        "english_gloss": "environment; surroundings",
    },
    {
        "surface": "技術",
        "reading": "ぎじゅつ",
        "jlpt_level": "N3",
        "part_of_speech": "noun",
        "english_gloss": "technology; engineering; skill",
    },
    {
        "surface": "影響",
        "reading": "えいきょう",
        "jlpt_level": "N3",
        "part_of_speech": "noun / suru-verb",
        "english_gloss": "influence; effect; impact",
    },
    {
        "surface": "効率",
        "reading": "こうりつ",
        "jlpt_level": "N3",
        "part_of_speech": "noun",
        "english_gloss": "efficiency",
    },
    {
        "surface": "改善",
        "reading": "かいぜん",
        "jlpt_level": "N3",
        "part_of_speech": "noun / suru-verb",
        "english_gloss": "continuous improvement; betterment",
    },
    # N2 pre-advanced vocabulary
    {
        "surface": "持続可能",
        "reading": "じぞくかのう",
        "jlpt_level": "N2",
        "part_of_speech": "na-adjective",
        "english_gloss": "sustainable",
    },
    {
        "surface": "導入",
        "reading": "どうにゅう",
        "jlpt_level": "N2",
        "part_of_speech": "noun / suru-verb",
        "english_gloss": "introduction; adoption; implementation",
    },
    {
        "surface": "需要",
        "reading": "じゅよう",
        "jlpt_level": "N2",
        "part_of_speech": "noun",
        "english_gloss": "demand; request",
    },
    {
        "surface": "推進",
        "reading": "すいしん",
        "jlpt_level": "N2",
        "part_of_speech": "noun / suru-verb",
        "english_gloss": "promotion; driving forward",
    },
    # N1 advanced vocabulary
    {
        "surface": "抜本的",
        "reading": "ばっぽんてき",
        "jlpt_level": "N1",
        "part_of_speech": "na-adjective",
        "english_gloss": "drastic; fundamental; sweeping",
    },
    {
        "surface": "不可欠",
        "reading": "ふかけつ",
        "jlpt_level": "N1",
        "part_of_speech": "na-adjective / noun",
        "english_gloss": "indispensable; essential",
    },
)

# Curated JLPT N5-N1 grammar corpus for vector/keyword search fallback.
SEED_JLPT_GRAMMAR: tuple[dict[str, Any], ...] = (
    {
        "grammar_id": "jlpt-n5-te-kara",
        "pattern": "〜てから",
        "match_tokens": ["てから", "してから", "読んでから", "食べてから"],
        "jlpt_level": "N5",
        "formation_rule": "Verb (て-form) + から",
        "english_meaning": "after doing ~; since doing ~",
        "japanese_example": "毎日図書館で勉強してから、電車で帰ります。",
        "english_translation": "Every day, after studying at the library, I go home by train.",
    },
    {
        "grammar_id": "jlpt-n4-you-ni-naru",
        "pattern": "〜ようになる",
        "match_tokens": ["ようになる", "ようになった", "ようになります", "話せるように"],
        "jlpt_level": "N4",
        "formation_rule": "Verb (Dictionary / Potential / ない-form) + ようになる",
        "english_meaning": "to reach the point where ~; to come to be that ~",
        "japanese_example": "毎日練習した結果、日本語が話せるようになった。",
        "english_translation": "As a result of practicing every day, I came to be able to speak Japanese.",
    },
    {
        "grammar_id": "jlpt-n3-ni-tsurete",
        "pattern": "〜につれて",
        "match_tokens": ["につれて", "につれ"],
        "jlpt_level": "N3",
        "formation_rule": "Verb (Dictionary form) / Noun + につれて",
        "english_meaning": "as ~; in proportion to ~; with ~",
        "japanese_example": "再生可能エネルギーの技術が進むにつれて、導入の効率も改善している。",
        "english_translation": "As renewable energy technology advances, adoption efficiency is also improving.",
    },
    {
        "grammar_id": "jlpt-n3-おかげで",
        "pattern": "〜おかげで",
        "match_tokens": ["おかげで", "おかげだ"],
        "jlpt_level": "N3",
        "formation_rule": "Verb / Adjective / Noun + の + おかげで",
        "english_meaning": "thanks to ~; owing to ~ (positive outcome)",
        "japanese_example": "新しい技術のおかげで、環境への影響を減らすことができた。",
        "english_translation": "Thanks to the new technology, we were able to reduce the impact on the environment.",
    },
    {
        "grammar_id": "jlpt-n2-ni-tomonai",
        "pattern": "〜に伴い（〜にともない）",
        "match_tokens": ["に伴い", "にともない", "に伴って", "にともなって"],
        "jlpt_level": "N2",
        "formation_rule": "Noun / Verb (Dictionary form) + に伴い / に伴って",
        "english_meaning": "accompanying ~; alongside ~; consequent upon ~",
        "japanese_example": "持続可能な社会への需要が増えるに伴い、抜本的な改革が推進されている。",
        "english_translation": "As demand for a sustainable society grows, fundamental reforms are being promoted.",
    },
    {
        "grammar_id": "jlpt-n1-nakushite-wa",
        "pattern": "〜なくしては",
        "match_tokens": ["なくしては", "なくして"],
        "jlpt_level": "N1",
        "formation_rule": "Noun / Verb (Dictionary form) + なくしては + Negative predicate",
        "english_meaning": "without ~ (it is impossible to ~)",
        "japanese_example": "抜本的な技術の改善なくしては、持続可能な発展は不可欠な目標を達成できない。",
        "english_translation": "Without fundamental technological improvement, sustainable development cannot achieve its essential goals.",
    },
)

# Registry of background asyncio tasks launched for memory consolidation.
_BACKGROUND_MEMORY_TASKS: list[asyncio.Task[dict[str, Any]]] = []


def estimate_token_count(text: str) -> int:
    """Estimate LLM token count for mixed Japanese/English text (approx. 1 token per 2 chars)."""
    if not text:
        return 0
    return max(1, (len(text) + 1) // 2)


def compact_conversation_history(
    messages: list[dict[str, str]],
    max_token_budget: int = 600,
    sliding_window_turns: int = 4,
) -> list[dict[str, str]]:
    """Compact conversation history using sliding-window retention, summarization, and token budgets.

    Preserves system instructions, summarizes older turns outside ``sliding_window_turns``
    into a single compact digest, and truncates if necessary so total estimated tokens stay
    within ``max_token_budget``.

    Args:
        messages: Ordered list of ``{"role": ..., "content": ...}`` conversation turns.
        max_token_budget: Maximum total estimated token budget allowed after compaction.
        sliding_window_turns: Number of most recent conversational turns to keep verbatim.

    Returns:
        Compacted list of PII-scrubbed message dicts within ``max_token_budget``.
    """
    if not messages:
        return []

    sanitized_messages = [
        {
            "role": str(msg.get("role", "user")),
            "content": scrub_pii(str(msg.get("content", ""))),
        }
        for msg in messages
    ]

    total_tokens = sum(estimate_token_count(m["content"]) for m in sanitized_messages)
    if total_tokens <= max_token_budget and len(sanitized_messages) <= sliding_window_turns + 1:
        return sanitized_messages

    system_turns = [m for m in sanitized_messages if m["role"] == "system"]
    dialogue_turns = [m for m in sanitized_messages if m["role"] != "system"]

    recent_turns = (
        dialogue_turns[-sliding_window_turns:]
        if sliding_window_turns > 0
        else []
    )
    older_turns = (
        dialogue_turns[:-sliding_window_turns]
        if sliding_window_turns > 0 and len(dialogue_turns) > sliding_window_turns
        else ([] if sliding_window_turns > 0 else dialogue_turns)
    )

    compacted: list[dict[str, str]] = list(system_turns)

    if older_turns:
        snippets = [
            f"{turn['role']}: {turn['content'][:60].strip()}"
            for turn in older_turns[-6:]
        ]
        summary_text = (
            f"[COMPACTED_HISTORY_SUMMARY turns={len(older_turns)}] "
            + " | ".join(snippets)
        )
        compacted.append({"role": "system", "content": summary_text})

    compacted.extend(recent_turns)

    # Enforce strict token budget ceiling by trimming content lengths if still over budget.
    while sum(estimate_token_count(m["content"]) for m in compacted) > max_token_budget:
        if len(compacted) > 2 and compacted[0]["role"] != "system":
            compacted.pop(0)
        else:
            longest_idx = max(
                range(len(compacted)),
                key=lambda idx: len(compacted[idx]["content"]),
            )
            current_text = compacted[longest_idx]["content"]
            if len(current_text) <= 32:
                break
            compacted[longest_idx] = {
                "role": compacted[longest_idx]["role"],
                "content": current_text[: max(32, len(current_text) // 2)] + "...[truncated]",
            }

    return compacted


def configure_vertex_context_cache(
    model_name: str = "gemini-2.5-flash",
    ttl_seconds: int = 3600,
    project_id: Optional[str] = None,
    location: str = "us-central1",
) -> dict[str, Any]:
    """Configure Vertex AI Context Caching for the static JLPT reference corpus.

    Uses ``google.genai.Client().caches.create(...)`` when Vertex AI credentials are
    available and returns a deterministic cache descriptor when running offline.
    """
    corpus_payload = json.dumps(
        {
            "vocabulary_count": len(SEED_JLPT_VOCABULARY),
            "grammar_count": len(SEED_JLPT_GRAMMAR),
            "vocabulary": SEED_JLPT_VOCABULARY,
            "grammar": SEED_JLPT_GRAMMAR,
        },
        ensure_ascii=False,
    )
    resolved_project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    force_offline = os.environ.get("NIHONGO_AGENT_FORCE_OFFLINE", "").lower() in {
        "1",
        "true",
        "yes",
    }

    if genai is not None and genai_types is not None and resolved_project and not force_offline:
        try:
            client = genai.Client(vertexai=True, project=resolved_project, location=location)
            cache = client.caches.create(
                model=model_name,
                config=genai_types.CreateCachedContentConfig(
                    display_name="nihongo-studyops-jlpt-reference-corpus",
                    system_instruction=(
                        "Cached JLPT N5-N1 vocabulary and grammar reference corpus for "
                        "low-latency study-pack generation."
                    ),
                    contents=[corpus_payload],
                    ttl=f"{ttl_seconds}s",
                ),
            )
            return {
                "cache_name": getattr(cache, "name", "vertex-cache-created"),
                "backend": "vertex_ai_context_cache",
                "model": model_name,
                "ttl_seconds": ttl_seconds,
            }
        except Exception:
            pass

    return {
        "cache_name": "local-context-cache://nihongo-studyops-jlpt-corpus-v1",
        "backend": "offline_context_cache_fallback",
        "model": model_name,
        "ttl_seconds": ttl_seconds,
        "cached_tokens_estimate": estimate_token_count(corpus_payload),
    }


class PersistentLearnerMemoryStore:
    """Manages persistent learner vocabulary, SRS decks, grammar retrieval, and turn history.

    Connects to Cloud Firestore and Vertex AI Search (Discovery Engine) when configured
    and automatically falls back to a local SQLite database for offline execution.
    """

    def __init__(
        self,
        sqlite_path: str = ":memory:",
        project_id: Optional[str] = None,
        location: str = "global",
        data_store_id: Optional[str] = None,
    ) -> None:
        self.sqlite_path = sqlite_path
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.data_store_id = data_store_id or os.environ.get("VERTEX_SEARCH_DATA_STORE_ID")
        self._conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._firestore_client: Optional[Any] = None
        self._search_client: Optional[Any] = None
        self._init_sqlite_schema_and_seed()

    def _init_sqlite_schema_and_seed(self) -> None:
        """Create SQLite tables and populate default JLPT vocabulary and grammar entries."""
        cursor = self._conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS jlpt_lexicon (
                surface TEXT PRIMARY KEY,
                reading TEXT NOT NULL,
                jlpt_level TEXT NOT NULL,
                part_of_speech TEXT NOT NULL,
                english_gloss TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jlpt_grammar_corpus (
                grammar_id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                match_tokens_json TEXT NOT NULL,
                jlpt_level TEXT NOT NULL,
                formation_rule TEXT NOT NULL,
                english_meaning TEXT NOT NULL,
                japanese_example TEXT NOT NULL,
                english_translation TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS learner_mastered_vocab (
                learner_id TEXT NOT NULL,
                surface TEXT NOT NULL,
                jlpt_level TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (learner_id, surface)
            );

            CREATE TABLE IF NOT EXISTS learner_srs_flashcards (
                learner_id TEXT NOT NULL,
                deck_name TEXT NOT NULL,
                card_id TEXT NOT NULL,
                front_japanese TEXT NOT NULL,
                back_reading_and_meaning TEXT NOT NULL,
                example_bilingual TEXT NOT NULL,
                jlpt_level TEXT NOT NULL,
                ease_factor REAL NOT NULL,
                interval_days INTEGER NOT NULL,
                next_review_iso TEXT NOT NULL,
                anki_tsv_row TEXT NOT NULL,
                PRIMARY KEY (learner_id, deck_name, card_id)
            );

            CREATE TABLE IF NOT EXISTS session_memory_consolidations (
                consolidation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                consolidated_at TEXT NOT NULL
            );
            """
        )

        for item in SEED_JLPT_VOCABULARY:
            cursor.execute(
                """
                INSERT OR IGNORE INTO jlpt_lexicon
                (surface, reading, jlpt_level, part_of_speech, english_gloss)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    item["surface"],
                    item["reading"],
                    item["jlpt_level"],
                    item["part_of_speech"],
                    item["english_gloss"],
                ),
            )

        for g_item in SEED_JLPT_GRAMMAR:
            cursor.execute(
                """
                INSERT OR IGNORE INTO jlpt_grammar_corpus
                (grammar_id, pattern, match_tokens_json, jlpt_level, formation_rule,
                 english_meaning, japanese_example, english_translation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    g_item["grammar_id"],
                    g_item["pattern"],
                    json.dumps(g_item["match_tokens"], ensure_ascii=False),
                    g_item["jlpt_level"],
                    g_item["formation_rule"],
                    g_item["english_meaning"],
                    g_item["japanese_example"],
                    g_item["english_translation"],
                ),
            )

        # Seed default mastered N5 words for 'default_learner' to demonstrate i+1 filtering.
        now_iso = datetime.now(timezone.utc).isoformat()
        for basic_word in ("学生", "毎日"):
            cursor.execute(
                """
                INSERT OR IGNORE INTO learner_mastered_vocab
                (learner_id, surface, jlpt_level, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                ("default_learner", basic_word, "N5", now_iso),
            )

        self._conn.commit()

    def _get_firestore_client(self) -> Optional[Any]:
        """Lazily initialize Cloud Firestore client when online and configured."""
        if firestore is None or not self.project_id:
            return None
        if os.environ.get("NIHONGO_AGENT_FORCE_OFFLINE", "").lower() in {"1", "true", "yes"}:
            return None
        if self._firestore_client is None:
            try:
                self._firestore_client = firestore.Client(project=self.project_id)
            except Exception:
                self._firestore_client = None
        return self._firestore_client

    def _get_search_client(self) -> Optional[Any]:
        """Lazily initialize Vertex AI Search ``SearchServiceClient`` when online."""
        if discoveryengine_v1 is None or not self.project_id or not self.data_store_id:
            return None
        if os.environ.get("NIHONGO_AGENT_FORCE_OFFLINE", "").lower() in {"1", "true", "yes"}:
            return None
        if self._search_client is None:
            try:
                self._search_client = discoveryengine_v1.SearchServiceClient()
            except Exception:
                self._search_client = None
        return self._search_client

    def get_mastered_words(self, learner_id: str) -> set[str]:
        """Return the set of Japanese vocabulary surface forms already mastered by ``learner_id``."""
        rows = self._conn.execute(
            "SELECT surface FROM learner_mastered_vocab WHERE learner_id = ?",
            (learner_id,),
        ).fetchall()
        return {str(row["surface"]) for row in rows}

    def mark_words_mastered(
        self,
        learner_id: str,
        words: list[tuple[str, str]],
    ) -> int:
        """Record ``(surface, jlpt_level)`` pairs as mastered for ``learner_id``."""
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.cursor()
        for surface, jlpt_level in words:
            cursor.execute(
                """
                INSERT OR REPLACE INTO learner_mastered_vocab
                (learner_id, surface, jlpt_level, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (learner_id, scrub_pii(surface), jlpt_level, now_iso),
            )
        self._conn.commit()
        return len(words)

    def extract_candidate_vocabulary(
        self,
        japanese_text: str,
        target_jlpt_level: JLPTLevel,
        learner_id: str,
        max_new_words: int = 8,
    ) -> tuple[list[VocabularyItem], list[str]]:
        """Match vocabulary in ``japanese_text`` against the JLPT lexicon, filtering mastered words."""
        mastered = self.get_mastered_words(learner_id)
        rows = self._conn.execute(
            "SELECT surface, reading, jlpt_level, part_of_speech, english_gloss FROM jlpt_lexicon"
        ).fetchall()

        target_rank = JLPT_DIFFICULTY_RANK.get(target_jlpt_level, 3)
        matched_unknown: list[ tuple[int, VocabularyItem] ] = []
        skipped_mastered: list[str] = []

        for row in rows:
            surface = str(row["surface"])
            if surface not in japanese_text:
                continue
            if surface in mastered:
                skipped_mastered.append(surface)
                continue

            item_level: JLPTLevel = row["jlpt_level"]
            item_rank = JLPT_DIFFICULTY_RANK.get(item_level, 3)
            # Prioritize words closest to the learner's target JLPT level.
            distance = abs(item_rank - target_rank)
            matched_unknown.append(
                (
                    distance,
                    VocabularyItem(
                        surface=surface,
                        reading=str(row["reading"]),
                        jlpt_level=item_level,
                        part_of_speech=str(row["part_of_speech"]),
                        english_gloss=str(row["english_gloss"]),
                        context_sentence=scrub_pii(japanese_text[:200]),
                    ),
                )
            )

        matched_unknown.sort(key=lambda pair: (pair[0], pair[1].surface))
        selected = [item for _, item in matched_unknown[:max_new_words]]
        return selected, skipped_mastered

    def search_grammar_corpus(
        self,
        japanese_text: str,
        target_jlpt_level: JLPTLevel = "N3",
        top_k: int = 4,
    ) -> tuple[list[GrammarPointMatch], str]:
        """Search Vertex AI Search (or local SQLite vector/corpus fallback) for JLPT grammar rules."""
        search_client = self._get_search_client()
        if search_client is not None and self.project_id and self.data_store_id:
            try:
                serving_config = (
                    f"projects/{self.project_id}/locations/{self.location}/collections/"
                    f"default_collection/dataStores/{self.data_store_id}/servingConfigs/default_search"
                )
                request = discoveryengine_v1.SearchRequest(
                    serving_config=serving_config,
                    query=japanese_text,
                    page_size=top_k,
                )
                response = search_client.search(request=request)
                cloud_matches: list[GrammarPointMatch] = []
                for result in response.results:
                    data = dict(result.document.struct_data)
                    cloud_matches.append(
                        GrammarPointMatch(
                            grammar_id=str(data.get("grammar_id", result.id)),
                            pattern=str(data.get("pattern", "〜")),
                            jlpt_level=data.get("jlpt_level", target_jlpt_level),
                            formation_rule=str(data.get("formation_rule", "Standard connection")),
                            english_meaning=str(data.get("english_meaning", "")),
                            japanese_example=str(data.get("japanese_example", "")),
                            english_translation=str(data.get("english_translation", "")),
                            relevance_score=0.95,
                        )
                    )
                if cloud_matches:
                    return cloud_matches[:top_k], "vertex_ai_search"
            except Exception:
                pass

        rows = self._conn.execute("SELECT * FROM jlpt_grammar_corpus").fetchall()
        target_rank = JLPT_DIFFICULTY_RANK.get(target_jlpt_level, 3)
        scored_matches: list[GrammarPointMatch] = []

        for row in rows:
            tokens: list[str] = json.loads(row["match_tokens_json"])
            token_hit = any(tok in japanese_text for tok in tokens)
            level = str(row["jlpt_level"])
            level_distance = abs(JLPT_DIFFICULTY_RANK.get(level, 3) - target_rank)

            if token_hit:
                score = round(max(0.65, 1.0 - 0.05 * level_distance), 2)
            elif level == target_jlpt_level:
                score = 0.60
            else:
                continue

            scored_matches.append(
                GrammarPointMatch(
                    grammar_id=str(row["grammar_id"]),
                    pattern=str(row["pattern"]),
                    jlpt_level=level,  # type: ignore[arg-type]
                    formation_rule=str(row["formation_rule"]),
                    english_meaning=str(row["english_meaning"]),
                    japanese_example=str(row["japanese_example"]),
                    english_translation=str(row["english_translation"]),
                    relevance_score=score,
                )
            )

        if not scored_matches and rows:
            first = rows[0]
            scored_matches.append(
                GrammarPointMatch(
                    grammar_id=str(first["grammar_id"]),
                    pattern=str(first["pattern"]),
                    jlpt_level=str(first["jlpt_level"]),  # type: ignore[arg-type]
                    formation_rule=str(first["formation_rule"]),
                    english_meaning=str(first["english_meaning"]),
                    japanese_example=str(first["japanese_example"]),
                    english_translation=str(first["english_translation"]),
                    relevance_score=0.50,
                )
            )

        scored_matches.sort(key=lambda item: item.relevance_score, reverse=True)
        return scored_matches[:top_k], "local_sqlite_vector_fallback"

    def save_flashcards(
        self,
        learner_id: str,
        deck_name: str,
        cards: list[FlashcardItem],
    ) -> str:
        """Persist scheduled flashcards to Cloud Firestore (when configured) and SQLite."""
        backend = "sqlite_fallback"
        fs_client = self._get_firestore_client()
        if fs_client is not None:
            try:
                batch = fs_client.batch()
                for card in cards:
                    doc_ref = (
                        fs_client.collection("learners")
                        .document(learner_id)
                        .collection("decks")
                        .document(deck_name)
                        .collection("cards")
                        .document(card.card_id)
                    )
                    batch.set(doc_ref, scrub_pii(card.model_dump()))
                batch.commit()
                backend = "cloud_firestore"
            except Exception:
                backend = "sqlite_fallback"

        cursor = self._conn.cursor()
        for card in cards:
            cursor.execute(
                """
                INSERT OR REPLACE INTO learner_srs_flashcards
                (learner_id, deck_name, card_id, front_japanese, back_reading_and_meaning,
                 example_bilingual, jlpt_level, ease_factor, interval_days, next_review_iso, anki_tsv_row)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    learner_id,
                    deck_name,
                    card.card_id,
                    scrub_pii(card.front_japanese),
                    scrub_pii(card.back_reading_and_meaning),
                    scrub_pii(card.example_bilingual),
                    card.jlpt_level,
                    card.ease_factor,
                    card.interval_days,
                    card.next_review_iso,
                    scrub_pii(card.anki_tsv_row),
                ),
            )
        self._conn.commit()
        return backend

    def get_deck_flashcards(self, learner_id: str, deck_name: str = "JLPT_Study_Pack") -> list[dict[str, Any]]:
        """Retrieve all persisted flashcards for ``learner_id`` and ``deck_name``."""
        rows = self._conn.execute(
            """
            SELECT * FROM learner_srs_flashcards
            WHERE learner_id = ? AND deck_name = ?
            ORDER BY card_id
            """,
            (learner_id, deck_name),
        ).fetchall()
        return [dict(row) for row in rows]

    def purge_learner_deck(self, learner_id: str, deck_name: str) -> tuple[int, int]:
        """Delete all flashcards in ``deck_name`` and reset custom mastered vocabulary for ``learner_id``."""
        cursor = self._conn.cursor()
        cards_deleted = cursor.execute(
            "DELETE FROM learner_srs_flashcards WHERE learner_id = ? AND deck_name = ?",
            (learner_id, deck_name),
        ).rowcount
        vocab_deleted = cursor.execute(
            "DELETE FROM learner_mastered_vocab WHERE learner_id = ?",
            (learner_id,),
        ).rowcount
        self._conn.commit()
        return max(0, cards_deleted), max(0, vocab_deleted)

    def record_consolidation(self, learner_id: str, summary_text: str) -> int:
        """Persist a PII-scrubbed memory consolidation digest into SQLite."""
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO session_memory_consolidations (learner_id, summary_text, consolidated_at)
            VALUES (?, ?, ?)
            """,
            (learner_id, scrub_pii(summary_text), now_iso),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)


_DEFAULT_MEMORY_STORE = PersistentLearnerMemoryStore()


def get_default_memory_store() -> PersistentLearnerMemoryStore:
    """Return the process-wide default ``PersistentLearnerMemoryStore`` instance."""
    return _DEFAULT_MEMORY_STORE


async def consolidate_and_persist_memory_async(
    learner_id: str,
    extracted_words: list[str],
    target_jlpt_level: str,
    store: Optional[PersistentLearnerMemoryStore] = None,
) -> dict[str, Any]:
    """Asynchronously consolidate newly encountered vocabulary into long-term learner memory."""
    active_store = store or _DEFAULT_MEMORY_STORE
    with trace_span(
        "memory.consolidate_async",
        attributes={"learner.id": learner_id, "jlpt.level": target_jlpt_level},
    ):
        await asyncio.sleep(0.005)
        digest = (
            f"Learner '{learner_id}' studied {len(extracted_words)} {target_jlpt_level} items: "
            + ", ".join(extracted_words)
        )
        consolidation_id = active_store.record_consolidation(
            learner_id=learner_id,
            summary_text=digest,
        )
        return {
            "status": "CONSOLIDATED",
            "consolidation_id": consolidation_id,
            "learner_id": learner_id,
            "items_consolidated": len(extracted_words),
        }


def schedule_background_memory_consolidation(
    learner_id: str,
    extracted_words: list[str],
    target_jlpt_level: str,
    store: Optional[PersistentLearnerMemoryStore] = None,
) -> bool:
    """Schedule ``consolidate_and_persist_memory_async`` non-blockingly via ``asyncio.create_task``."""
    coro = consolidate_and_persist_memory_async(
        learner_id=learner_id,
        extracted_words=extracted_words,
        target_jlpt_level=target_jlpt_level,
        store=store,
    )
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro, name=f"consolidate-memory-{learner_id}")
        _BACKGROUND_MEMORY_TASKS.append(task)
        return True
    except RuntimeError:
        # When invoked outside an active event loop, execute on a dedicated runner loop.
        asyncio.run(coro)
        return True


async def await_pending_memory_tasks() -> list[dict[str, Any]]:
    """Await and drain any pending background memory consolidation tasks."""
    if not _BACKGROUND_MEMORY_TASKS:
        return []
    pending = list(_BACKGROUND_MEMORY_TASKS)
    _BACKGROUND_MEMORY_TASKS.clear()
    results = await asyncio.gather(*pending, return_exceptions=False)
    return list(results)
