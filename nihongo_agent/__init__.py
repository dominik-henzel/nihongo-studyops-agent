"""Nihongo StudyOps Agent package.

Provides strict Pydantic v2 contracts, Secret Manager resolution, structured
observability, persistent session memory, guardrails, and multi-agent ADK
orchestration for generating Japanese JLPT Anki study packs.
"""

from __future__ import annotations

from . import schemas, secrets

__all__ = ["schemas", "secrets"]
