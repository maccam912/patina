"""Basic tests for Patina package."""

import pytest
import asyncio
from uuid import uuid4


class TestConfig:
    """Tests for configuration module."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from patina.config import PatinaConfig
        
        config = PatinaConfig()
        
        assert config.context_total_tokens == 128000
        assert config.context_output_reserved == 4000
        assert config.persona_block_limit == 2000
        assert config.human_block_limit == 2000
        assert config.consolidation_enabled is True
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from patina.config import PatinaConfig
        
        config = PatinaConfig(
            database_url="sqlite+aiosqlite:///test.db",
            context_total_tokens=64000,
        )
        
        assert "test.db" in config.database_url
        assert config.context_total_tokens == 64000


class TestDatabase:
    """Tests for database module."""
    
    @pytest.mark.asyncio
    async def test_database_connect(self):
        """Test database connection."""
        from patina.config import PatinaConfig
        from patina.database.connection import Database
        
        config = PatinaConfig(database_url="sqlite+aiosqlite:///:memory:")
        db = Database(config)
        
        await db.connect()
        assert db._engine is not None
        
        await db.disconnect()
    
    @pytest.mark.asyncio
    async def test_database_init_schema(self):
        """Test schema initialization."""
        from patina.config import PatinaConfig
        from patina.database.connection import Database
        
        config = PatinaConfig(database_url="sqlite+aiosqlite:///:memory:")
        db = Database(config)
        
        await db.connect()
        await db.init_schema()
        
        # Check that tables exist
        tables = await db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = [t["name"] for t in tables]
        
        assert "tenants" in table_names
        assert "agents" in table_names
        assert "conversations" in table_names
        assert "messages" in table_names
        assert "memory_blocks" in table_names
        
        await db.disconnect()
    
    @pytest.mark.asyncio
    async def test_basic_crud(self):
        """Test basic CRUD operations."""
        from patina.config import PatinaConfig
        from patina.database.connection import Database
        from datetime import datetime, UTC
        
        config = PatinaConfig(database_url="sqlite+aiosqlite:///:memory:")
        db = Database(config)
        
        await db.connect()
        await db.init_schema()
        
        # Create tenant
        tenant_id = uuid4()
        await db.execute(
            """
            INSERT INTO tenants (id, name, created_at)
            VALUES (:id, :name, :created_at)
            """,
            {
                "id": str(tenant_id),
                "name": "test_tenant",
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        
        # Read tenant
        result = await db.fetch_one(
            "SELECT * FROM tenants WHERE id = :id",
            {"id": str(tenant_id)},
        )
        
        assert result is not None
        assert result["name"] == "test_tenant"
        
        await db.disconnect()


class TestModels:
    """Tests for data models."""
    
    def test_memory_block_model(self):
        """Test MemoryBlock model."""
        from patina.database.models import MemoryBlock, MemoryBlockType
        
        block = MemoryBlock(
            tenant_id=uuid4(),
            agent_id=uuid4(),
            block_type=MemoryBlockType.FACT,
            content="User prefers dark mode",
        )
        
        assert block.block_type == MemoryBlockType.FACT
        assert block.content == "User prefers dark mode"
        assert block.importance == 0.5  # default
        assert block.confidence == 0.8  # default in schema is 0.8
    
    def test_message_model(self):
        """Test Message model."""
        from patina.database.models import Message, MessageRole
        
        msg = Message(
            tenant_id=uuid4(),
            conversation_id=uuid4(),
            role=MessageRole.USER,
            content="Hello!",
        )
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"



class TestMemory:
    """Tests for memory system."""
    
    @pytest.mark.asyncio
    async def test_memory_store(self):
        """Test storing and retrieving memories."""
        from patina.config import PatinaConfig
        from patina.database.connection import Database
        from patina.memory.semantic import MemoryBlockStore
        from patina.database.models import MemoryBlockType
        
        config = PatinaConfig(database_url="sqlite+aiosqlite:///:memory:")
        db = Database(config)
        
        await db.connect()
        await db.init_schema()
        
        store = MemoryBlockStore(db)
        
        tenant_id = uuid4()
        agent_id = uuid4()
        
        # Create tenant and agent first
        from datetime import datetime
        await db.execute(
            "INSERT INTO tenants (id, name, created_at) VALUES (:id, :name, :now)",
            {"id": str(tenant_id), "name": "test", "now": datetime.utcnow().isoformat()},
        )
        await db.execute(
            """INSERT INTO agents (id, tenant_id, name, system_prompt, created_at, updated_at)
               VALUES (:id, :tid, :name, :prompt, :now, :now)""",
            {
                "id": str(agent_id),
                "tid": str(tenant_id),
                "name": "test_agent",
                "prompt": "test",
                "now": datetime.utcnow().isoformat(),
            },
            tenant_id,
        )
        
        # Store memory
        block = await store.create(
            tenant_id=tenant_id,
            agent_id=agent_id,
            block_type=MemoryBlockType.PREFERENCE,
            content="User likes pizza",
            importance=0.7,
        )
        
        assert block.content == "User likes pizza"
        assert block.importance == 0.7
        
        # Retrieve memories
        memories = await store.get_for_agent(
            agent_id=agent_id,
            tenant_id=tenant_id,
        )
        
        assert len(memories) == 1
        assert memories[0].content == "User likes pizza"
        
        await db.disconnect()


class TestLLMRouter:
    """Tests for LLM router."""
    
    def test_mock_mode(self):
        """Test mock mode activation without API keys."""
        import os
        
        # Remove any API keys from env
        old_keys = {}
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
            if key in os.environ:
                old_keys[key] = os.environ.pop(key)
        
        try:
            from patina.config import PatinaConfig
            from patina.llm.router import LLMRouter
            
            config = PatinaConfig(llm_api_key=None)
            router = LLMRouter(config)
            
            # Access chat to trigger initialization
            _ = router.chat
            
            assert router._mock_mode is True
        finally:
            # Restore keys
            for key, value in old_keys.items():
                os.environ[key] = value
    
    @pytest.mark.asyncio
    async def test_mock_completion(self):
        """Test mock completion."""
        from patina.config import PatinaConfig
        from patina.llm.router import LLMRouter
        
        config = PatinaConfig(llm_api_key=None)
        router = LLMRouter(config)
        router._mock_mode = True
        
        response = await router.completion(
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert response.choices[0].message.content is not None
        assert "Mock response" in response.choices[0].message.content


class TestDecay:
    """Tests for memory decay."""
    
    def test_retrievability_calculation(self):
        """Test Ebbinghaus forgetting curve."""
        from patina.consolidation.decay import MemoryDecayManager
        from datetime import datetime, timedelta
        
        # Mock db not needed for this test
        manager = MemoryDecayManager(db=None)
        
        now = datetime.utcnow()
        
        # Just accessed - high retrievability
        R = manager.calculate_retrievability(now, stability=1.0, now=now)
        assert abs(R - 1.0) < 0.01
        
        # 1 day ago with stability 1 - about 37%
        yesterday = now - timedelta(days=1)
        R = manager.calculate_retrievability(yesterday, stability=1.0, now=now)
        assert 0.3 < R < 0.4
        
        # Higher stability means slower decay
        R_stable = manager.calculate_retrievability(yesterday, stability=10.0, now=now)
        assert R_stable > R
    
    def test_next_review_calculation(self):
        """Test optimal review interval."""
        from patina.consolidation.decay import MemoryDecayManager
        
        manager = MemoryDecayManager(db=None)
        
        # With stability 1, to maintain 90% retrievability
        days = manager.calculate_next_review(stability=1.0, target_retrievability=0.9)
        assert 0.1 < days < 0.2  # About 0.1 days
        
        # Higher stability means longer interval
        days_stable = manager.calculate_next_review(stability=10.0, target_retrievability=0.9)
        assert days_stable > days


class TestTools:
    """Tests for tool definitions."""
    
    def test_memory_tools(self):
        """Test memory tool definitions."""
        from patina.llm.tools import get_memory_tools, MEMORY_TOOLS
        
        tools = get_memory_tools()
        
        assert len(tools) == 2
        
        names = [t["function"]["name"] for t in tools]
        assert "memory_search" in names
        assert "memory_store" in names
    
    def test_extended_tools(self):
        """Test extended tool set."""
        from patina.llm.tools import get_memory_tools
        
        tools = get_memory_tools(include_update=True)
        
        assert len(tools) == 3
        names = [t["function"]["name"] for t in tools]
        assert "memory_update" in names
