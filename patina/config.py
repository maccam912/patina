"""Configuration management for Patina."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class PatinaConfig(BaseSettings):
    """Configuration for Patina agent framework."""
    
    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///patina.db",
        description="Database connection URL (PostgreSQL or SQLite for dev)",
    )
    
    # LLM Configuration
    llm_api_key: Optional[str] = Field(default=None, description="Primary LLM API key")
    llm_model: str = Field(
        default="anthropic/claude-sonnet-4-20250514",
        description="Default model for agent responses",
    )
    llm_fast_model: str = Field(
        default="anthropic/claude-3-5-haiku-20241022",
        description="Fast model for summarization",
    )
    llm_fallback_model: str = Field(
        default="openai/gpt-4o",
        description="Fallback model",
    )
    
    # Context Window
    context_total_tokens: int = Field(default=128000, description="Total context window size")
    context_output_reserved: int = Field(default=4000, description="Reserved for output")
    context_warning_tokens: int = Field(default=90000, description="Warning threshold")
    context_flush_tokens: int = Field(default=100000, description="Flush threshold")
    
    # Memory Block Limits
    persona_block_limit: int = Field(default=2000, description="Max chars for persona block")
    human_block_limit: int = Field(default=2000, description="Max chars for human block")
    
    # Consolidation
    consolidation_enabled: bool = Field(default=True, description="Enable consolidation jobs")
    daily_consolidation_hour: int = Field(default=1, description="Hour for daily consolidation")
    daily_consolidation_minute: int = Field(default=30, description="Minute for daily consolidation")
    weekly_consolidation_day: int = Field(default=6, description="Day for weekly (0=Mon, 6=Sun)")
    weekly_consolidation_hour: int = Field(default=3, description="Hour for weekly consolidation")
    monthly_consolidation_day: int = Field(default=1, description="Day for monthly consolidation")
    monthly_consolidation_hour: int = Field(default=4, description="Hour for monthly consolidation")
    
    # Forgetting Curve
    decay_base_stability: float = Field(default=1.0, description="Base stability in days")
    decay_stability_multiplier: float = Field(default=2.5, description="Stability multiplier on access")
    decay_archive_threshold: float = Field(default=0.1, description="Archive when R < this")
    decay_prune_threshold: float = Field(default=0.05, description="Prune when R < this")
    
    # Embedding
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model for embeddings",
    )
    embedding_dimensions: int = Field(default=1536, description="Embedding vector dimensions")
    
    model_config = {"env_prefix": "PATINA_", "env_file": ".env"}
