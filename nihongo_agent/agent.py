"""Multi-agent ADK orchestration and strategic model routing for Nihongo StudyOps.

Defines a modular multi-agent topology built with Google Agent Development Kit (ADK):
- **Strategic Model Routing (``select_model_for_task``)**:
  Routes high-throughput extraction, grammar lookup, and SRS scheduling tasks to
  ``gemini-2.5-flash`` and routes multi-agent coordination, diagnostic curriculum
  planning, and HITL governance to ``gemini-2.5-pro``.
- **3-Stage Sequential Pipeline (``study_pack_sequential_pipeline``)**:
  A ``SequentialAgent`` chaining ``vocabulary_miner_agent``,
  ``grammar_analyzer_agent``, and ``srs_deck_builder_agent``.
- **Diagnostic Planner & HITL Gate (``curriculum_planner_agent``)**:
  An ``LlmAgent`` on ``gemini-2.5-pro`` equipped with ``before_tool_hitl_callback``
  for high-stakes deck lifecycle operations.
- **Root Coordinator (``root_agent`` / ``nihongo_studyops_coordinator``)**:
  Top-level ``LlmAgent`` wrapped in an ADK ``App`` with ``EventsCompactionConfig``
  and ``StudyOpsPolicyPlugin``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .guardrails import (
    StudyOpsPolicyPlugin,
    after_model_self_eval_callback,
    before_model_guardrail_callback,
    before_tool_hitl_callback,
)
from .prompts import (
    COORDINATOR_CONSTITUTION_PROMPT,
    CURRICULUM_PLANNER_CONSTITUTION_PROMPT,
    GRAMMAR_ANALYZER_CONSTITUTION_PROMPT,
    SRS_DECK_BUILDER_CONSTITUTION_PROMPT,
    VOCABULARY_MINER_CONSTITUTION_PROMPT,
)
from .tools import (
    extract_unknown_jlpt_vocabulary_from_text,
    purge_and_reset_learner_srs_deck_critical,
    schedule_spaced_repetition_flashcard_batch,
    search_jlpt_grammar_vector_store,
)

try:
    from google.adk.agents import LlmAgent, SequentialAgent
    from google.adk.apps.app import App, EventsCompactionConfig
    from google.adk.sessions import DatabaseSessionService, InMemorySessionService

    ADK_AVAILABLE = True
except ImportError:  # pragma: no cover - lightweight fallback if google-adk is not installed
    ADK_AVAILABLE = False
    DatabaseSessionService = None  # type: ignore[assignment]
    InMemorySessionService = None  # type: ignore[assignment]

    class LlmAgent:  # type: ignore[no-redef]
        """Fallback LlmAgent descriptor preserving full ADK configuration attributes."""

        def __init__(
            self,
            *,
            name: str,
            model: str,
            instruction: str = "",
            description: str = "",
            tools: Optional[list[Callable[..., Any]]] = None,
            sub_agents: Optional[list[Any]] = None,
            output_key: Optional[str] = None,
            before_model_callback: Optional[Callable[..., Any]] = None,
            after_model_callback: Optional[Callable[..., Any]] = None,
            before_tool_callback: Optional[Callable[..., Any]] = None,
        ) -> None:
            self.name = name
            self.model = model
            self.instruction = instruction
            self.description = description
            self.tools = list(tools or [])
            self.sub_agents = list(sub_agents or [])
            self.output_key = output_key
            self.before_model_callback = before_model_callback
            self.after_model_callback = after_model_callback
            self.before_tool_callback = before_tool_callback

    class SequentialAgent:  # type: ignore[no-redef]
        """Fallback SequentialAgent descriptor preserving ordered sub-agent pipeline."""

        def __init__(
            self,
            *,
            name: str,
            description: str = "",
            sub_agents: Optional[list[Any]] = None,
        ) -> None:
            self.name = name
            self.description = description
            self.sub_agents = list(sub_agents or [])

    class EventsCompactionConfig:  # type: ignore[no-redef]
        """Fallback EventsCompactionConfig descriptor."""

        def __init__(self, compaction_interval: int = 4, overlap_size: int = 1) -> None:
            self.compaction_interval = compaction_interval
            self.overlap_size = overlap_size

    class App:  # type: ignore[no-redef]
        """Fallback ADK App container."""

        def __init__(
            self,
            *,
            name: str,
            root_agent: Any,
            plugins: Optional[list[Any]] = None,
            events_compaction_config: Optional[EventsCompactionConfig] = None,
        ) -> None:
            self.name = name
            self.root_agent = root_agent
            self.plugins = list(plugins or [])
            self.events_compaction_config = events_compaction_config


FAST_EXTRACTION_MODEL = "gemini-2.5-flash"
DEEP_REASONING_MODEL = "gemini-2.5-pro"

_FAST_TASK_TYPES: frozenset[str] = frozenset(
    {
        "vocabulary_extraction",
        "grammar_analysis",
        "srs_scheduling",
        "fast_lookup",
        "flashcard_formatting",
    }
)
_PRO_TASK_TYPES: frozenset[str] = frozenset(
    {
        "coordination",
        "curriculum_planning",
        "diagnostic_reasoning",
        "hitl_deck_governance",
        "multi_step_strategy",
    }
)


def select_model_for_task(task_type: str, complexity_score: float = 0.3) -> str:
    """Strategically route an agent task to ``gemini-2.5-flash`` or ``gemini-2.5-pro``.

    Args:
        task_type: Semantic task category (e.g., ``'vocabulary_extraction'`` or
            ``'curriculum_planning'``).
        complexity_score: Normalized reasoning complexity between 0.0 and 1.0.

    Returns:
        Model identifier string (``'gemini-2.5-flash'`` or ``'gemini-2.5-pro'``).
    """
    normalized_task = (task_type or "").strip().lower()
    if normalized_task in _PRO_TASK_TYPES or complexity_score >= 0.7:
        return DEEP_REASONING_MODEL
    if normalized_task in _FAST_TASK_TYPES:
        return FAST_EXTRACTION_MODEL
    return FAST_EXTRACTION_MODEL


def create_persistent_session_service(sqlite_db_url: str = "sqlite:///nihongo_sessions.sqlite3") -> Any:
    """Create a persistent ADK ``DatabaseSessionService`` with ``InMemorySessionService`` fallback."""
    if DatabaseSessionService is not None:
        try:
            return DatabaseSessionService(db_url=sqlite_db_url)
        except Exception:
            pass
    if InMemorySessionService is not None:
        return InMemorySessionService()
    return {"backend": "sqlite_fallback", "db_url": sqlite_db_url}


# Stage 1 Sub-Agent: Mines unknown i+1 JLPT vocabulary on gemini-2.5-flash.
vocabulary_miner_agent = LlmAgent(
    name="vocabulary_miner_agent",
    model=select_model_for_task("vocabulary_extraction", complexity_score=0.2),
    description=(
        "Extracts unknown i+1 Japanese vocabulary from raw text, filters out words "
        "already mastered by the learner, and triggers async memory consolidation."
    ),
    instruction=VOCABULARY_MINER_CONSTITUTION_PROMPT,
    tools=[extract_unknown_jlpt_vocabulary_from_text],
    output_key="extracted_vocabulary",
    before_model_callback=before_model_guardrail_callback,
    after_model_callback=after_model_self_eval_callback,
)

# Stage 2 Sub-Agent: Grounds JLPT grammar explanations against Vertex AI Search / vector store.
grammar_analyzer_agent = LlmAgent(
    name="grammar_analyzer_agent",
    model=select_model_for_task("grammar_analysis", complexity_score=0.3),
    description=(
        "Retrieves verified JLPT N5-N1 grammar formation rules and bilingual examples "
        "from the persistent grammar vector corpus."
    ),
    instruction=GRAMMAR_ANALYZER_CONSTITUTION_PROMPT,
    tools=[search_jlpt_grammar_vector_store],
    output_key="grammar_analysis",
    before_model_callback=before_model_guardrail_callback,
    after_model_callback=after_model_self_eval_callback,
)

# Stage 3 Sub-Agent: Computes SM-2 review intervals and persists Anki flashcards.
srs_deck_builder_agent = LlmAgent(
    name="srs_deck_builder_agent",
    model=select_model_for_task("srs_scheduling", complexity_score=0.25),
    description=(
        "Schedules spaced-repetition flashcards using SM-2 intervals, exports Anki TSV "
        "rows, and persists cards to Cloud Firestore or SQLite."
    ),
    instruction=SRS_DECK_BUILDER_CONSTITUTION_PROMPT,
    tools=[schedule_spaced_repetition_flashcard_batch],
    output_key="scheduled_deck_summary",
    before_model_callback=before_model_guardrail_callback,
    after_model_callback=after_model_self_eval_callback,
)

# Sequential Pipeline orchestrating the 3 study-pack stages in deterministic order.
study_pack_sequential_pipeline = SequentialAgent(
    name="study_pack_sequential_pipeline",
    description=(
        "Autonomous 3-step Japanese study-pack pipeline chaining Vocabulary Miner, "
        "Grammar Analyzer, and SRS Deck Builder agents."
    ),
    sub_agents=[
        vocabulary_miner_agent,
        grammar_analyzer_agent,
        srs_deck_builder_agent,
    ],
)

# Specialist Planner Agent on gemini-2.5-pro with Human-in-the-Loop protection on deck resets.
curriculum_planner_agent = LlmAgent(
    name="curriculum_planner_agent",
    model=select_model_for_task("curriculum_planning", complexity_score=0.9),
    description=(
        "Designs multi-week JLPT study roadmaps on gemini-2.5-pro and governs "
        "human-confirmed spaced-repetition deck resets."
    ),
    instruction=CURRICULUM_PLANNER_CONSTITUTION_PROMPT,
    tools=[purge_and_reset_learner_srs_deck_critical],
    before_model_callback=before_model_guardrail_callback,
    after_model_callback=after_model_self_eval_callback,
    before_tool_callback=before_tool_hitl_callback,
)

# Root Coordinator Agent exported for ADK CLI ('adk run', 'adk web', 'adk deploy').
root_agent = LlmAgent(
    name="nihongo_studyops_coordinator",
    model=select_model_for_task("coordination", complexity_score=0.85),
    description=(
        "Root coordinator that routes Japanese reading passages to the 3-stage "
        "study_pack_sequential_pipeline and routes diagnostic planning or deck resets "
        "to curriculum_planner_agent."
    ),
    instruction=COORDINATOR_CONSTITUTION_PROMPT,
    sub_agents=[
        study_pack_sequential_pipeline,
        curriculum_planner_agent,
    ],
    before_model_callback=before_model_guardrail_callback,
    after_model_callback=after_model_self_eval_callback,
    before_tool_callback=before_tool_hitl_callback,
)

# ADK App container with native event history compaction and policy plugin registration.
app = App(
    name="nihongo_agent",
    root_agent=root_agent,
    plugins=[StudyOpsPolicyPlugin()],
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=4,
        overlap_size=1,
    ),
)
