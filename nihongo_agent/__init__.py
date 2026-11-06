"""Nihongo StudyOps Agent package.

Exports ``root_agent`` and ``app`` at package level for Google ADK CLI discovery
(``adk run nihongo_agent``, ``adk web``, ``adk eval``, ``adk deploy agent_engine``).
"""

from __future__ import annotations

from . import agent, guardrails, memory, observability, prompts, schemas, secrets, tools
from .agent import app, root_agent

__all__ = [
    "agent",
    "app",
    "guardrails",
    "memory",
    "observability",
    "prompts",
    "root_agent",
    "schemas",
    "secrets",
    "tools",
]
