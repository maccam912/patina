"""Patina: Biologically-inspired AI agent memory framework."""

from patina.config import PatinaConfig
from patina.agent.core import PatinaAgent
from patina.database.connection import Database
from patina.memory.manager import MemoryManager
from patina.llm.router import LLMRouter
from patina.consolidation.consolidator import MemoryConsolidator
from patina.scheduling.jobs import ConsolidationScheduler

__version__ = "0.1.0"

__all__ = [
    "PatinaAgent",
    "PatinaConfig",
    "Database",
    "MemoryManager",
    "LLMRouter",
    "MemoryConsolidator",
    "ConsolidationScheduler",
]
