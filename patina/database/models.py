"""Database models for Patina memory framework."""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role types."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class JournalEntryType(str, Enum):
    """Journal entry types."""
    DAILY_REFLECTION = "daily_reflection"
    WEEKLY_SYNTHESIS = "weekly_synthesis"
    MONTHLY_INTEGRATION = "monthly_integration"
    MILESTONE = "milestone"
    INSIGHT = "insight"


class MemoryBlockType(str, Enum):
    """Memory block types."""
    FACT = "fact"
    PREFERENCE = "preference"
    BELIEF = "belief"
    SKILL = "skill"
    RELATIONSHIP = "relationship"
    SUMMARY = "summary"
    PERSONALITY_TRAIT = "personality_trait"


class ToolImplementationType(str, Enum):
    """Tool implementation types."""
    BUILTIN = "builtin"
    WEBHOOK = "webhook"
    PYTHON = "python"
    SUBAGENT = "subagent"


class SubAgentRelationshipType(str, Enum):
    """Sub-agent relationship types."""
    SPECIALIST = "specialist"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    WRITER = "writer"


class TaskType(str, Enum):
    """Scheduled task types."""
    DAILY_CONSOLIDATION = "daily_consolidation"
    WEEKLY_SYNTHESIS = "weekly_synthesis"
    MONTHLY_INTEGRATION = "monthly_integration"
    MEMORY_DECAY = "memory_decay"
    GARBAGE_COLLECTION = "garbage_collection"
    EMBEDDING_SYNC = "embedding_sync"


class TaskStatus(str, Enum):
    """Scheduled task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


# ============================================================================
# CORE MODELS
# ============================================================================


class Tenant(BaseModel):
    """Multi-tenant organization."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    settings: dict = Field(default_factory=dict)
    timezone: str = "UTC"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Agent(BaseModel):
    """AI Agent configuration."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    name: str
    system_prompt: str
    model_config_data: dict = Field(
        default_factory=lambda: {"model": "anthropic/claude-sonnet-4-20250514", "temperature": 0.7},
        alias="model_config",
    )
    
    # Core memory blocks (Letta-style editable context)
    persona_block: str = ""
    persona_block_limit: int = 2000
    human_block: str = ""
    human_block_limit: int = 2000
    
    # Memory pressure thresholds
    context_warning_tokens: int = 90000
    context_flush_tokens: int = 100000
    
    # Consolidation settings
    consolidation_enabled: bool = True
    consolidation_hour: int = 2
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}


class Conversation(BaseModel):
    """Conversation session."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    agent_id: UUID
    user_id: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    summary_token_count: int = 0
    message_count: int = 0
    metadata: dict = Field(default_factory=dict)
    is_archived: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Message(BaseModel):
    """Individual message in a conversation."""
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    tenant_id: UUID
    role: MessageRole
    content: str
    
    # Tool calling support
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    
    # Embeddings (stored separately in production)
    embedding: Optional[List[float]] = None
    
    # Memory metadata
    token_count: Optional[int] = None
    importance_score: float = 0.5
    is_summarized: bool = False
    metadata: dict = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JournalEntry(BaseModel):
    """Consolidated reflection entry."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    agent_id: UUID
    user_id: Optional[str] = None
    
    entry_type: JournalEntryType
    content: str
    extracted_facts: List[dict] = Field(default_factory=list)
    emotional_valence: Optional[float] = None  # -1 to 1
    significance_score: float = 0.5
    
    source_conversation_ids: List[UUID] = Field(default_factory=list)
    source_message_range: Optional[dict] = None
    embedding: Optional[List[float]] = None
    consolidation_run_id: Optional[UUID] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryBlock(BaseModel):
    """Semantic memory storage unit."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    agent_id: UUID
    user_id: Optional[str] = None
    
    block_type: MemoryBlockType
    content: str
    embedding: Optional[List[float]] = None
    
    # Spaced repetition / forgetting curve parameters
    stability: float = 1.0  # S in R = e^(-t/S)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 1
    next_review_at: Optional[datetime] = None
    
    # Importance and lifecycle
    importance: float = 0.5
    confidence: float = 0.8
    source_journal_ids: List[UUID] = Field(default_factory=list)
    superseded_by: Optional[UUID] = None
    
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class Tool(BaseModel):
    """Tool definition for agent use."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    name: str
    description: str
    parameters_schema: dict
    implementation_type: ToolImplementationType
    implementation_config: dict
    requires_confirmation: bool = False
    is_enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SubAgent(BaseModel):
    """Sub-agent relationship definition."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    parent_agent_id: UUID
    child_agent_id: UUID
    relationship_type: SubAgentRelationshipType
    delegation_prompt: Optional[str] = None
    share_memory: bool = False
    share_context: bool = True
    max_tokens: int = 4000
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentTool(BaseModel):
    """Agent-tool association."""
    agent_id: UUID
    tool_id: UUID


class ScheduledTask(BaseModel):
    """Scheduled consolidation task."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    agent_id: Optional[UUID] = None
    
    task_type: TaskType
    schedule_cron: Optional[str] = None
    schedule_timezone: str = "UTC"
    next_run_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    
    status: TaskStatus = TaskStatus.PENDING
    last_result: Optional[dict] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    config: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
