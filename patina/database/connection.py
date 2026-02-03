"""Database connection and transaction management."""

from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Any
from uuid import UUID
import json

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

from patina.config import PatinaConfig


class Database:
    """Async database connection manager with tenant isolation support."""
    
    def __init__(self, config: PatinaConfig):
        self.config = config
        self._engine = None
        self._session_factory = None
        self._current_tenant_id: Optional[UUID] = None
    
    async def connect(self) -> None:
        """Initialize database connection pool."""
        # SQLite doesn't support pool_size and max_overflow
        is_sqlite = "sqlite" in self.config.database_url.lower()
        
        engine_kwargs = {
            "echo": False,
            "json_serializer": lambda obj: json.dumps(obj, default=str),
        }
        
        if not is_sqlite:
            engine_kwargs["pool_size"] = 5
            engine_kwargs["max_overflow"] = 10
        
        self._engine = create_async_engine(
            self.config.database_url,
            **engine_kwargs,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    
    async def disconnect(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
    
    @asynccontextmanager
    async def session(self, tenant_id: Optional[UUID] = None) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with optional tenant context.
        
        Args:
            tenant_id: If provided, sets RLS context for row-level security.
        
        Yields:
            AsyncSession for database operations.
        """
        if not self._session_factory:
            await self.connect()
        
        async with self._session_factory() as session:
            # Set tenant context for RLS policies (PostgreSQL only)
            if tenant_id and "postgresql" in self.config.database_url:
                await session.execute(
                    text(f"SET app.tenant_id = '{tenant_id}'")
                )
            
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    @asynccontextmanager
    async def transaction(self, tenant_id: Optional[UUID] = None) -> AsyncGenerator[AsyncSession, None]:
        """Explicit transaction context (same as session but clearer intent)."""
        async with self.session(tenant_id) as session:
            yield session
    
    async def execute(
        self,
        query: str,
        params: Optional[dict] = None,
        tenant_id: Optional[UUID] = None,
    ) -> Any:
        """Execute a raw SQL query."""
        async with self.session(tenant_id) as session:
            result = await session.execute(text(query), params or {})
            return result
    
    async def fetch_one(
        self,
        query: str,
        params: Optional[dict] = None,
        tenant_id: Optional[UUID] = None,
    ) -> Optional[dict]:
        """Fetch a single row as dict."""
        result = await self.execute(query, params, tenant_id)
        row = result.fetchone()
        if row:
            return dict(row._mapping)
        return None
    
    async def fetch_all(
        self,
        query: str,
        params: Optional[dict] = None,
        tenant_id: Optional[UUID] = None,
    ) -> list[dict]:
        """Fetch all rows as list of dicts."""
        result = await self.execute(query, params, tenant_id)
        return [dict(row._mapping) for row in result.fetchall()]
    
    async def init_schema(self) -> None:
        """Initialize database schema (for development/testing)."""
        schema_sql = """
        -- Tenants
        CREATE TABLE IF NOT EXISTS tenants (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            settings TEXT DEFAULT '{}',
            timezone TEXT DEFAULT 'UTC',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Agents
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            model_config TEXT DEFAULT '{}',
            persona_block TEXT DEFAULT '',
            persona_block_limit INTEGER DEFAULT 2000,
            human_block TEXT DEFAULT '',
            human_block_limit INTEGER DEFAULT 2000,
            context_warning_tokens INTEGER DEFAULT 90000,
            context_flush_tokens INTEGER DEFAULT 100000,
            consolidation_enabled INTEGER DEFAULT 1,
            consolidation_hour INTEGER DEFAULT 2,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Conversations
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            agent_id TEXT NOT NULL REFERENCES agents(id),
            user_id TEXT,
            title TEXT,
            summary TEXT,
            summary_token_count INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            is_archived INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Messages
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id),
            tenant_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
            content TEXT NOT NULL,
            tool_calls TEXT,
            tool_call_id TEXT,
            embedding TEXT,
            token_count INTEGER,
            importance_score REAL DEFAULT 0.5,
            is_summarized INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Journal Entries
        CREATE TABLE IF NOT EXISTS journal_entries (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            agent_id TEXT NOT NULL REFERENCES agents(id),
            user_id TEXT,
            entry_type TEXT NOT NULL,
            content TEXT NOT NULL,
            extracted_facts TEXT DEFAULT '[]',
            emotional_valence REAL,
            significance_score REAL DEFAULT 0.5,
            source_conversation_ids TEXT DEFAULT '[]',
            source_message_range TEXT,
            embedding TEXT,
            consolidation_run_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Memory Blocks
        CREATE TABLE IF NOT EXISTS memory_blocks (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            agent_id TEXT NOT NULL REFERENCES agents(id),
            user_id TEXT,
            block_type TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT,
            stability REAL DEFAULT 1.0,
            last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 1,
            next_review_at TIMESTAMP,
            importance REAL DEFAULT 0.5,
            confidence REAL DEFAULT 0.8,
            source_journal_ids TEXT DEFAULT '[]',
            superseded_by TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        );
        
        -- Tools
        CREATE TABLE IF NOT EXISTS tools (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            parameters_schema TEXT NOT NULL,
            implementation_type TEXT NOT NULL,
            implementation_config TEXT NOT NULL,
            requires_confirmation INTEGER DEFAULT 0,
            is_enabled INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (tenant_id, name)
        );
        
        -- Sub-agents
        CREATE TABLE IF NOT EXISTS sub_agents (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            parent_agent_id TEXT NOT NULL REFERENCES agents(id),
            child_agent_id TEXT NOT NULL REFERENCES agents(id),
            relationship_type TEXT NOT NULL,
            delegation_prompt TEXT,
            share_memory INTEGER DEFAULT 0,
            share_context INTEGER DEFAULT 1,
            max_tokens INTEGER DEFAULT 4000,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Agent-Tool association
        CREATE TABLE IF NOT EXISTS agent_tools (
            agent_id TEXT NOT NULL REFERENCES agents(id),
            tool_id TEXT NOT NULL REFERENCES tools(id),
            PRIMARY KEY (agent_id, tool_id)
        );
        
        -- Scheduled Tasks
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            agent_id TEXT REFERENCES agents(id),
            task_type TEXT NOT NULL,
            schedule_cron TEXT,
            schedule_timezone TEXT DEFAULT 'UTC',
            next_run_at TIMESTAMP,
            last_run_at TIMESTAMP,
            status TEXT DEFAULT 'pending',
            last_result TEXT,
            error_message TEXT,
            retry_count INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3,
            config TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes (SQLite compatible)
        CREATE INDEX IF NOT EXISTS idx_agents_tenant ON agents(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_tenant_agent ON conversations(tenant_id, agent_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(tenant_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_messages_tenant_time ON messages(tenant_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_memory_blocks_agent_type ON memory_blocks(agent_id, block_type);
        CREATE INDEX IF NOT EXISTS idx_memory_blocks_user ON memory_blocks(agent_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_memory_blocks_importance ON memory_blocks(agent_id, importance);
        CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_next_run ON scheduled_tasks(next_run_at);
        """
        
        async with self.session() as session:
            for statement in schema_sql.split(";"):
                statement = statement.strip()
                if statement:
                    await session.execute(text(statement))
