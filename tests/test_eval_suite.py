"""Automated offline evaluation and regression test suite for Nihongo StudyOps Agent.

Executes end-to-end golden dataset scenarios (`tests/golden_dataset.json`) and
verifies all 19 architectural invariants across Tool & Interface Design, Context &
Memory, Orchestration & Logic, Observability & Tracing, and Infrastructure & CI/CD
without requiring live cloud credentials.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nihongo_agent import (
    agent,
    app,
    guardrails,
    memory,
    observability,
    prompts,
    root_agent,
    schemas,
    secrets,
    tools,
)

GOLDEN_DATASET_PATH = REPO_ROOT / "tests" / "golden_dataset.json"


class NihongoStudyOpsEvaluationSuite(unittest.TestCase):
    """Comprehensive evaluation suite testing golden scenarios and AgentOps architecture."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["NIHONGO_AGENT_FORCE_OFFLINE"] = "true"
        with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as handle:
            cls.golden_data: dict[str, Any] = json.load(handle)

    def setUp(self) -> None:
        observability.clear_structured_log_entries()
        observability.clear_recorded_spans()

    # -------------------------------------------------------------------------
    # Category 1: Tool & Interface Design
    # -------------------------------------------------------------------------

    def test_1_1_and_1_2_descriptive_tool_names_and_comprehensive_docstrings(self) -> None:
        """Verify all 4 tools use descriptive names and document purpose + all parameters."""
        expected_tools = {
            "extract_unknown_jlpt_vocabulary_from_text": tools.extract_unknown_jlpt_vocabulary_from_text,
            "search_jlpt_grammar_vector_store": tools.search_jlpt_grammar_vector_store,
            "schedule_spaced_repetition_flashcard_batch": tools.schedule_spaced_repetition_flashcard_batch,
            "purge_and_reset_learner_srs_deck_critical": tools.purge_and_reset_learner_srs_deck_critical,
        }
        for name, func in expected_tools.items():
            self.assertTrue(callable(func), f"Tool {name} must be callable.")
            self.assertGreaterEqual(len(name), 25, f"Tool name {name} should be descriptive.")
            doc = inspect.getdoc(func) or ""
            self.assertIn("Args:", doc, f"Tool {name} docstring must include an 'Args:' section.")
            self.assertIn("Returns:", doc, f"Tool {name} docstring must include a 'Returns:' section.")
            sig = inspect.signature(func)
            for param_name in sig.parameters:
                self.assertIn(
                    param_name,
                    doc,
                    f"Parameter '{param_name}' must be documented in {name} docstring.",
                )

    def test_1_3_strict_pydantic_v2_input_and_output_schemas(self) -> None:
        """Verify strict Pydantic v2 schemas reject extra fields and constrain tool inputs/outputs."""
        schema_classes = [
            schemas.VocabularyItem,
            schemas.VocabularyExtractionInput,
            schemas.VocabularyExtractionOutput,
            schemas.GrammarPointMatch,
            schemas.GrammarSearchInput,
            schemas.GrammarSearchOutput,
            schemas.FlashcardItem,
            schemas.FlashcardBatchInput,
            schemas.FlashcardBatchOutput,
            schemas.ResetLearnerDeckInput,
            schemas.ResetLearnerDeckOutput,
        ]
        for model_cls in schema_classes:
            self.assertEqual(
                model_cls.model_config.get("extra"),
                "forbid",
                f"{model_cls.__name__} must enforce ConfigDict(extra='forbid').",
            )

        # Verify hallucinated extra field is rejected and translated into guided recovery.
        bad_extra_res = tools.extract_unknown_jlpt_vocabulary_from_text(
            japanese_text="再生可能エネルギーの技術を勉強する。",
            target_jlpt_level="N3",
            extra_fields={"hallucinated_argument": "unexpected"},
        )
        self.assertEqual(bad_extra_res["status"], "VALIDATION_ERROR")
        self.assertIn("RECOVERY INSTRUCTION", bad_extra_res["recovery_instructions"])

    def test_1_4_guided_error_handling_returns_recovery_instructions(self) -> None:
        """Verify invalid tool inputs return structured recovery instructions instead of crashing."""
        invalid_case = next(
            c
            for c in self.golden_data["adversarial_and_edge_cases"]
            if c["case_id"] == "invalid_ascii_only_input"
        )
        res = tools.extract_unknown_jlpt_vocabulary_from_text(
            japanese_text=invalid_case["japanese_text"],
            target_jlpt_level=invalid_case["target_jlpt_level"],
        )
        self.assertEqual(res["status"], invalid_case["expected_status"])
        self.assertIsNotNone(res["recovery_instructions"])
        self.assertIn(invalid_case["expected_recovery_substring"], res["recovery_instructions"])

        empty_batch = tools.schedule_spaced_repetition_flashcard_batch(
            learner_id="default_learner",
            vocabulary_items=[],
            grammar_points=[],
        )
        self.assertEqual(empty_batch["status"], "VALIDATION_ERROR")
        self.assertIn("RECOVERY INSTRUCTION", empty_batch["recovery_instructions"])

    # -------------------------------------------------------------------------
    # Category 2: Context & Memory
    # -------------------------------------------------------------------------

    def test_2_1_robust_xml_system_constitutions(self) -> None:
        """Verify all agents define XML-structured <AGENT_CONSTITUTION> system instructions."""
        self.assertGreaterEqual(len(prompts.ALL_AGENT_PROMPTS), 5)
        for agent_name, prompt_text in prompts.ALL_AGENT_PROMPTS.items():
            for required_tag in (
                "<AGENT_CONSTITUTION>",
                "<PERSONA>",
                "<DOMAIN_KNOWLEDGE>",
                "<OPERATIONAL_CONSTRAINTS>",
                "<OUTPUT_PROTOCOL>",
            ):
                self.assertIn(
                    required_tag,
                    prompt_text,
                    f"Prompt for {agent_name} is missing {required_tag}.",
                )

    def test_2_2_history_compaction_and_vertex_context_caching(self) -> None:
        """Verify sliding-window history compaction, token truncation, and Vertex context caching."""
        bloated_history = [
            {"role": "system", "content": "System constitution for JLPT study pack agent."},
        ]
        for turn_idx in range(12):
            bloated_history.append(
                {
                    "role": "user" if turn_idx % 2 == 0 else "assistant",
                    "content": f"Turn {turn_idx}: 再生可能エネルギーの技術と環境への影響を詳しく説明します。" * 3,
                }
            )

        compacted = memory.compact_conversation_history(
            messages=bloated_history,
            max_token_budget=220,
            sliding_window_turns=3,
        )
        compacted_tokens = sum(memory.estimate_token_count(m["content"]) for m in compacted)
        self.assertLessEqual(compacted_tokens, 220)
        self.assertTrue(
            any("[COMPACTED_HISTORY_SUMMARY" in m["content"] for m in compacted),
            "Older turns must be summarized into a [COMPACTED_HISTORY_SUMMARY] digest.",
        )

        cache_meta = memory.configure_vertex_context_cache(model_name="gemini-2.5-flash")
        self.assertIn("cache_name", cache_meta)
        self.assertIsNotNone(app.events_compaction_config)

    def test_2_3_and_2_4_persistent_session_state_and_async_memory_operations(self) -> None:
        """Verify persistent SQLite/Firestore memory store and non-blocking asyncio consolidation."""
        temp_store = memory.PersistentLearnerMemoryStore(sqlite_path=":memory:")
        mastered_before = temp_store.get_mastered_words("default_learner")
        self.assertIn("学生", mastered_before)
        self.assertIn("毎日", mastered_before)

        async def _run_async_consolidation() -> list[dict[str, Any]]:
            scheduled = memory.schedule_background_memory_consolidation(
                learner_id="async_test_learner",
                extracted_words=["再生可能", "環境"],
                target_jlpt_level="N3",
                store=temp_store,
            )
            self.assertTrue(scheduled)
            return await memory.await_pending_memory_tasks()

        results = asyncio.run(_run_async_consolidation())
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "CONSOLIDATED")
        self.assertEqual(results[0]["items_consolidated"], 2)

    # -------------------------------------------------------------------------
    # Category 3: Orchestration & Logic
    # -------------------------------------------------------------------------

    def test_3_1_and_3_2_multi_agent_patterns_and_strategic_model_routing(self) -> None:
        """Verify Coordinator + SequentialAgent hierarchy and Flash vs. Pro model routing."""
        self.assertEqual(root_agent.name, "nihongo_studyops_coordinator")
        self.assertEqual(root_agent.model, "gemini-2.5-pro")
        sub_agent_names = [sa.name for sa in root_agent.sub_agents]
        self.assertIn("study_pack_sequential_pipeline", sub_agent_names)
        self.assertIn("curriculum_planner_agent", sub_agent_names)

        pipeline_stages = agent.study_pack_sequential_pipeline.sub_agents
        self.assertEqual(
            [stage.name for stage in pipeline_stages],
            [
                "vocabulary_miner_agent",
                "grammar_analyzer_agent",
                "srs_deck_builder_agent",
            ],
        )
        for fast_stage in pipeline_stages:
            self.assertEqual(fast_stage.model, "gemini-2.5-flash")

        self.assertEqual(agent.curriculum_planner_agent.model, "gemini-2.5-pro")
        self.assertEqual(agent.select_model_for_task("vocabulary_extraction"), "gemini-2.5-flash")
        self.assertEqual(agent.select_model_for_task("curriculum_planning", 0.9), "gemini-2.5-pro")

    def test_3_3_guardrails_and_self_evaluation_plugin(self) -> None:
        """Verify prompt-injection guardrails, self-evaluation checks, and ADK policy plugin."""
        injection_case = next(
            c
            for c in self.golden_data["adversarial_and_edge_cases"]
            if c["case_id"] == "prompt_injection_override"
        )
        decision = guardrails.evaluate_input_guardrails(injection_case["japanese_text"])
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.status, injection_case["expected_status"])
        self.assertEqual(decision.triggered_rule, injection_case["expected_triggered_rule"])

        bad_self_eval = guardrails.self_evaluate_study_pack_output(
            "English output only without any Kanji or Kana."
        )
        self.assertFalse(bad_self_eval.allowed)
        self.assertEqual(bad_self_eval.status, "SELF_EVAL_WARNING")

        good_self_eval = guardrails.self_evaluate_study_pack_output(
            "単語: 再生可能（さいせいかのう） - renewable"
        )
        self.assertTrue(good_self_eval.allowed)

    def test_3_4_human_in_the_loop_hook_blocks_unapproved_destructive_reset(self) -> None:
        """Verify before_tool_hitl_callback blocks unapproved deck resets and allows approved ones."""
        hitl_case = next(
            c
            for c in self.golden_data["adversarial_and_edge_cases"]
            if c["case_id"] == "hitl_unapproved_deck_reset"
        )
        blocked = tools.purge_and_reset_learner_srs_deck_critical(
            learner_id=hitl_case["learner_id"],
            deck_name=hitl_case["deck_name"],
            reset_reason=hitl_case["reset_reason"],
            human_confirmation_approved=False,
        )
        self.assertEqual(blocked["status"], "PENDING_HUMAN_APPROVAL")
        self.assertFalse(blocked["human_approved"])
        self.assertIn("human_confirmation_approved=True", blocked["recovery_instructions"])

        approved = tools.purge_and_reset_learner_srs_deck_critical(
            learner_id="temp_reset_learner",
            deck_name="Temp_Deck",
            reset_reason="Confirmed test cleanup after user approval.",
            human_confirmation_approved=True,
        )
        self.assertEqual(approved["status"], "SUCCESS")
        self.assertTrue(approved["human_approved"])

    # -------------------------------------------------------------------------
    # Category 4: Observability & Tracing
    # -------------------------------------------------------------------------

    def test_4_1_4_2_4_3_structured_json_logging_intent_outcome_and_otel_tracing(self) -> None:
        """Verify structured JSON logs (no print()), INTENT/OUTCOME phases, and OpenTelemetry spans."""
        result = tools.run_study_pack_pipeline_offline(
            japanese_text="再生可能エネルギーの技術が進むにつれて、環境への影響を減らす。",
            target_jlpt_level="N3",
            learner_id="default_learner",
        )
        self.assertEqual(result["status"], "SUCCESS")

        logs = observability.get_structured_log_entries()
        self.assertGreaterEqual(len(logs), 6)
        phases = {entry.get("phase") for entry in logs}
        self.assertIn("INTENT", phases)
        self.assertIn("OUTCOME", phases)

        outcome_entries = [e for e in logs if e.get("phase") == "OUTCOME"]
        for entry in outcome_entries:
            self.assertIsNotNone(entry.get("latency_ms"))
            self.assertIsNotNone(entry.get("trace_id"))

        spans = observability.get_recorded_spans()
        span_names = {s["name"] for s in spans}
        self.assertIn("pipeline.study_pack_sequential", span_names)
        self.assertIn("tool.extract_unknown_jlpt_vocabulary_from_text", span_names)
        self.assertIn("tool.search_jlpt_grammar_vector_store", span_names)
        self.assertIn("tool.schedule_spaced_repetition_flashcard_batch", span_names)

        # Ensure zero print() calls exist anywhere in nihongo_agent/*.py.
        pkg_dir = REPO_ROOT / "nihongo_agent"
        for py_file in pkg_dir.glob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotEqual(
                        node.func.id,
                        "print",
                        f"Forbidden print() call found in {py_file.name}; use structured logging.",
                    )

    def test_4_4_pii_redaction_scrubber(self) -> None:
        """Verify PII (email, phone, Japanese MyNumber, SSN, API keys) is redacted before storage."""
        pii_case = next(
            c
            for c in self.golden_data["adversarial_and_edge_cases"]
            if c["case_id"] == "pii_redaction_in_passage"
        )
        scrubbed = observability.scrub_pii(pii_case["raw_input"])
        for token in pii_case["expected_redacted_tokens"]:
            self.assertIn(token, scrubbed)
        for forbidden in pii_case["forbidden_raw_substrings"]:
            self.assertNotIn(forbidden, scrubbed)

    # -------------------------------------------------------------------------
    # Category 5: Infrastructure, CI/CD, Golden Dataset & Secret Management
    # -------------------------------------------------------------------------

    def test_5_1_golden_dataset_study_pack_scenarios_end_to_end(self) -> None:
        """Execute all study-pack scenarios from tests/golden_dataset.json end-to-end."""
        for scenario in self.golden_data["study_pack_scenarios"]:
            with self.subTest(scenario_id=scenario["scenario_id"]):
                output = tools.run_study_pack_pipeline_offline(
                    japanese_text=scenario["japanese_text"],
                    target_jlpt_level=scenario["target_jlpt_level"],
                    learner_id=scenario["learner_id"],
                )
                self.assertEqual(output["status"], "SUCCESS")

                extracted_surfaces = [
                    item["surface"] for item in output["vocabulary"]["extracted_vocabulary"]
                ]
                for expected_word in scenario["expected_extracted_surfaces"]:
                    self.assertIn(expected_word, extracted_surfaces)

                for mastered_word in scenario["expected_skipped_mastered"]:
                    self.assertIn(mastered_word, output["vocabulary"]["skipped_mastered_words"])
                    self.assertNotIn(mastered_word, extracted_surfaces)

                grammar_patterns = [
                    g["pattern"] for g in output["grammar"]["matched_grammar_points"]
                ]
                self.assertIn(scenario["expected_grammar_pattern"], grammar_patterns)
                self.assertGreaterEqual(
                    output["flashcards"]["scheduled_cards_count"],
                    scenario["min_scheduled_cards"],
                )
                self.assertIn("\t", output["flashcards"]["anki_tsv_export"])
                self.assertTrue(output["self_evaluation"]["allowed"])

    def test_5_2_terraform_iac_and_readme_cli_documentation(self) -> None:
        """Verify Terraform IaC files and CLI deployment documentation exist."""
        main_tf = (REPO_ROOT / "terraform" / "main.tf").read_text(encoding="utf-8")
        variables_tf = (REPO_ROOT / "terraform" / "variables.tf").read_text(encoding="utf-8")
        outputs_tf = (REPO_ROOT / "terraform" / "outputs.tf").read_text(encoding="utf-8")
        readme_md = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        for required_resource in (
            "google_project_service",
            "google_service_account",
            "google_firestore_database",
            "google_discovery_engine_data_store",
            "google_secret_manager_secret",
        ):
            self.assertIn(required_resource, main_tf)

        self.assertIn("variable \"project_id\"", variables_tf)
        self.assertIn("output \"agent_service_account_email\"", outputs_tf)
        for cli_doc in ("adk run", "adk web", "adk deploy", "agents-cli", "terraform"):
            self.assertIn(cli_doc, readme_md)

    def test_5_3_secure_secret_management_with_environment_fallback(self) -> None:
        """Verify SecretResolver resolves secrets via Secret Manager / env without hardcoded keys."""
        os.environ["TEST_STUDYOPS_SECRET_TOKEN"] = "ephemeral-test-token-value"
        try:
            resolved = secrets.get_secret("TEST_STUDYOPS_SECRET_TOKEN")
            self.assertEqual(resolved, "ephemeral-test-token-value")
        finally:
            os.environ.pop("TEST_STUDYOPS_SECRET_TOKEN", None)
            secrets.get_secret.cache_clear()

    def test_repository_privacy_and_git_identity_hygiene(self) -> None:
        """Ensure zero internal domains, hostnames, or course terms appear in files or git log."""
        forbidden_markers = [
            "".join(("@goo", "gle.com")),
            "".join(("c.goo", "glers.com")),
            "".join(("roam.corp.goo", "gle.com")),
            "".join(("arg", "olis")),
            "".join(("star", "fleet")),
            "".join(("noo", "gler")),
            "".join(("ge", "ap")),
            "".join(("l2", "00")),
        ]

        tracked_files = subprocess.check_output(
            ["git", "ls-files"], cwd=REPO_ROOT, text=True
        ).splitlines()
        for rel_path in tracked_files:
            file_path = REPO_ROOT / rel_path
            if not file_path.is_file():
                continue
            content_lower = file_path.read_text(encoding="utf-8", errors="ignore").lower()
            for marker in forbidden_markers:
                self.assertNotIn(
                    marker,
                    content_lower,
                    f"Forbidden marker '{marker}' found in tracked file {rel_path}.",
                )

        git_log_output = subprocess.check_output(
            ["git", "log", "--format=fuller"], cwd=REPO_ROOT, text=True
        ).lower()
        for marker in forbidden_markers:
            self.assertNotIn(
                marker,
                git_log_output,
                f"Forbidden marker '{marker}' found in git commit history.",
            )


if __name__ == "__main__":
    unittest.main()
