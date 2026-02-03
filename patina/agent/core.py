"""Core agent: Main PatinaAgent implementation."""

from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator
from uuid import UUID, uuid4
import json

from patina.config import PatinaConfig
from patina.database.connection import Database
from patina.database.models import (
    Tenant,
    Agent,
    Conversation,
    Message,
    MessageRole,
    MemoryBlockType,
)
from patina.memory.manager import MemoryManager
from patina.llm.router import LLMRouter
from patina.llm.tools import get_memory_tools, format_tool_result


class PatinaAgent:
    """Main agent class with memory-augmented conversations.
    
    PatinaAgent provides:
    - Conversation management with automatic memory retrieval
    - Tool calling for memory operations
    - Context window management with cognitive limits
    - Automatic importance scoring for messages
    
    Example:
        ```python
        agent = await PatinaAgent.create(
            config=config,
            name="Assistant",
            system_prompt="You are a helpful assistant.",
        )
        
        response = await agent.chat("Hello!")
        ```
    """
    
    def __init__(
        self,
        db: Database,
        config: PatinaConfig,
        agent_data: Dict,
        tenant_id: UUID,
        llm: LLMRouter,
        memory: MemoryManager,
    ):
        self.db = db
        self.config = config
        self._agent_data = agent_data
        self.tenant_id = tenant_id
        self.llm = llm
        self.memory = memory
        
        self._current_conversation: Optional[Conversation] = None
        self._user_id: Optional[str] = None
    
    @classmethod
    async def create(
        cls,
        config: Optional[PatinaConfig] = None,
        name: str = "Assistant",
        system_prompt: str = "You are a helpful AI assistant.",
        persona: str = "",
        tenant_name: str = "default",
        db: Optional[Database] = None,
    ) -> "PatinaAgent":
        """Create a new PatinaAgent.
        
        Args:
            config: Configuration (uses defaults if not provided)
            name: Agent name
            system_prompt: Base system prompt
            persona: Initial persona block
            tenant_name: Tenant name for multi-tenancy
            db: Optional existing database connection
            
        Returns:
            Initialized PatinaAgent
        """
        config = config or PatinaConfig()
        
        # Initialize database
        if db is None:
            db = Database(config)
            await db.connect()
            await db.init_schema()
        
        # Create or get tenant
        tenant_id = await cls._ensure_tenant(db, tenant_name)
        
        # Initialize LLM router
        llm = LLMRouter(config)
        
        # Create embedding function
        async def embedding_fn(text: str) -> List[float]:
            return await llm.embed(text)
        
        # Initialize memory manager
        memory = MemoryManager(
            db=db,
            config=config,
            embedding_fn=embedding_fn,
            llm_client=llm.chat,
        )
        
        # Create agent record
        agent_data = await cls._ensure_agent(
            db=db,
            tenant_id=tenant_id,
            name=name,
            system_prompt=system_prompt,
            persona=persona,
            config=config,
        )
        
        return cls(
            db=db,
            config=config,
            agent_data=agent_data,
            tenant_id=tenant_id,
            llm=llm,
            memory=memory,
        )
    
    @staticmethod
    async def _ensure_tenant(db: Database, name: str) -> UUID:
        """Get or create tenant."""
        row = await db.fetch_one(
            "SELECT id FROM tenants WHERE name = :name",
            {"name": name},
        )
        
        if row:
            return UUID(row["id"])
        
        tenant_id = uuid4()
        async with db.session() as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    INSERT INTO tenants (id, name, created_at)
                    VALUES (:id, :name, :created_at)
                """),
                {
                    "id": str(tenant_id),
                    "name": name,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
        return tenant_id
    
    @staticmethod
    async def _ensure_agent(
        db: Database,
        tenant_id: UUID,
        name: str,
        system_prompt: str,
        persona: str,
        config: PatinaConfig,
    ) -> Dict:
        """Get or create agent record."""
        row = await db.fetch_one(
            "SELECT * FROM agents WHERE tenant_id = :tenant_id AND name = :name",
            {"tenant_id": str(tenant_id), "name": name},
            tenant_id,
        )
        
        if row:
            return row
        
        agent_id = uuid4()
        now = datetime.utcnow().isoformat()
        
        agent_data = {
            "id": str(agent_id),
            "tenant_id": str(tenant_id),
            "name": name,
            "system_prompt": system_prompt,
            "model_config": json.dumps({
                "model": config.llm_model,
                "temperature": 0.7,
            }),
            "persona_block": persona,
            "persona_block_limit": config.persona_block_limit,
            "human_block": "",
            "human_block_limit": config.human_block_limit,
            "context_warning_tokens": config.context_warning_tokens,
            "context_flush_tokens": config.context_flush_tokens,
            "consolidation_enabled": 1 if config.consolidation_enabled else 0,
            "consolidation_hour": config.daily_consolidation_hour,
            "created_at": now,
            "updated_at": now,
        }
        
        async with db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    INSERT INTO agents 
                    (id, tenant_id, name, system_prompt, model_config,
                     persona_block, persona_block_limit, human_block, human_block_limit,
                     context_warning_tokens, context_flush_tokens,
                     consolidation_enabled, consolidation_hour, created_at, updated_at)
                    VALUES (:id, :tenant_id, :name, :system_prompt, :model_config,
                            :persona_block, :persona_block_limit, :human_block, :human_block_limit,
                            :context_warning_tokens, :context_flush_tokens,
                            :consolidation_enabled, :consolidation_hour, :created_at, :updated_at)
                """),
                agent_data,
            )
        
        return agent_data
    
    @property
    def agent_id(self) -> UUID:
        """Get agent ID."""
        return UUID(self._agent_data["id"])
    
    @property
    def name(self) -> str:
        """Get agent name."""
        return self._agent_data.get("name", "Assistant")
    
    @property
    def persona(self) -> str:
        """Get current persona block."""
        return self._agent_data.get("persona_block", "")
    
    @property
    def human_block(self) -> str:
        """Get current human/user block."""
        return self._agent_data.get("human_block", "")
    
    async def start_conversation(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Conversation:
        """Start a new conversation.
        
        Args:
            user_id: Optional user identifier
            title: Optional conversation title
            
        Returns:
            New conversation
        """
        self._user_id = user_id
        self._current_conversation = await self.memory.start_conversation(
            agent_id=self.agent_id,
            tenant_id=self.tenant_id,
            user_id=user_id,
            title=title,
        )
        return self._current_conversation
    
    async def chat(
        self,
        message: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        use_tools: bool = True,
    ) -> str:
        """Send a message and get a response.
        
        Args:
            message: User message
            user_id: Optional user identifier
            conversation_id: Optional existing conversation
            use_tools: Whether to enable memory tools
            
        Returns:
            Assistant response text
        """
        # Ensure conversation exists
        if conversation_id:
            self._current_conversation = await self.memory.get_conversation(
                conversation_id=conversation_id,
                tenant_id=self.tenant_id,
            )
        
        if not self._current_conversation:
            await self.start_conversation(user_id=user_id)
        
        self._user_id = user_id or self._user_id
        
        # Store user message
        await self.memory.add_message(
            conversation_id=self._current_conversation.id,
            tenant_id=self.tenant_id,
            role=MessageRole.USER,
            content=message,
        )
        
        # Build context
        context = await self.memory.build_context(
            agent={
                "id": self.agent_id,
                "system_prompt": self._agent_data.get("system_prompt", ""),
                "persona_block": self.persona,
                "human_block": self.human_block,
            },
            conversation_id=self._current_conversation.id,
            tenant_id=self.tenant_id,
            current_query=message,
            user_id=self._user_id,
        )
        
        # Get tools
        tools = get_memory_tools() if use_tools else None
        
        # Generate response
        response = await self._generate_response(context, tools)
        
        # Store assistant message
        await self.memory.add_message(
            conversation_id=self._current_conversation.id,
            tenant_id=self.tenant_id,
            role=MessageRole.ASSISTANT,
            content=response,
        )
        
        return response
    
    async def _generate_response(
        self,
        context: List[Dict],
        tools: Optional[List[Dict]] = None,
    ) -> str:
        """Generate response using LLM with optional tool handling."""
        response = await self.llm.completion(
            messages=context,
            model="default",
            tools=tools,
        )
        
        message = response.choices[0].message
        
        # Handle tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            return await self._handle_tool_calls(context, message, tools)
        
        return message.content or ""
    
    async def _handle_tool_calls(
        self,
        context: List[Dict],
        message: Any,
        tools: Optional[List[Dict]],
    ) -> str:
        """Execute tool calls and get final response."""
        # Add assistant message with tool calls
        context.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            result = await self._execute_tool(tool_call)
            context.append(format_tool_result(tool_call.id, result))
        
        # Get final response
        response = await self.llm.completion(
            messages=context,
            model="default",
            tools=tools,
        )
        
        return response.choices[0].message.content or ""
    
    async def _execute_tool(self, tool_call: Any) -> str:
        """Execute a single tool call."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return "Error: Invalid arguments"
        
        if name == "memory_search":
            return await self._tool_memory_search(args)
        elif name == "memory_store":
            return await self._tool_memory_store(args)
        elif name == "memory_update":
            return await self._tool_memory_update(args)
        else:
            return f"Error: Unknown tool {name}"
    
    async def _tool_memory_search(self, args: Dict) -> str:
        """Execute memory_search tool."""
        query = args.get("query", "")
        memory_types = args.get("memory_types")
        user_id = args.get("user_id", self._user_id)
        
        # Convert type strings to enums
        type_enums = None
        if memory_types:
            try:
                type_enums = [MemoryBlockType(t) for t in memory_types]
            except ValueError:
                pass
        
        results = await self.memory.search_memories(
            query=query,
            agent_id=self.agent_id,
            tenant_id=self.tenant_id,
            user_id=user_id,
            memory_types=type_enums,
            limit=5,
        )
        
        if not results:
            return "No relevant memories found."
        
        lines = ["Found memories:"]
        for r in results:
            mem = r["memory"]
            lines.append(f"- [{mem.block_type.value}] {mem.content[:150]}...")
        
        return "\n".join(lines)
    
    async def _tool_memory_store(self, args: Dict) -> str:
        """Execute memory_store tool."""
        content = args.get("content", "")
        memory_type = args.get("memory_type", "fact")
        importance = args.get("importance", 0.5)
        user_id = args.get("user_id", self._user_id)
        
        try:
            block_type = MemoryBlockType(memory_type)
        except ValueError:
            block_type = MemoryBlockType.FACT
        
        await self.memory.store_memory(
            agent_id=self.agent_id,
            tenant_id=self.tenant_id,
            content=content,
            block_type=block_type,
            user_id=user_id,
            importance=importance,
        )
        
        return f"Stored memory: {content[:100]}..."
    
    async def _tool_memory_update(self, args: Dict) -> str:
        """Execute memory_update tool."""
        old_content = args.get("old_content", "")
        new_content = args.get("new_content", "")
        
        # Search for matching memory
        results = await self.memory.search_memories(
            query=old_content,
            agent_id=self.agent_id,
            tenant_id=self.tenant_id,
            limit=1,
        )
        
        if not results:
            return "No matching memory found to update."
        
        old_memory = results[0]["memory"]
        
        # Create new memory
        new_memory = await self.memory.store_memory(
            agent_id=self.agent_id,
            tenant_id=self.tenant_id,
            content=new_content,
            block_type=old_memory.block_type,
            user_id=old_memory.user_id,
            importance=old_memory.importance,
        )
        
        # Supersede old memory
        await self.memory.memory_blocks.supersede(
            old_memory_id=old_memory.id,
            new_memory_id=new_memory.id,
            tenant_id=self.tenant_id,
        )
        
        return f"Updated memory: {new_content[:100]}..."
    
    async def update_persona(self, persona: str) -> None:
        """Update the agent's persona block."""
        self._agent_data["persona_block"] = persona[:self.config.persona_block_limit]
        
        async with self.db.session(self.tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("UPDATE agents SET persona_block = :persona, updated_at = :now WHERE id = :id"),
                {
                    "id": str(self.agent_id),
                    "persona": self._agent_data["persona_block"],
                    "now": datetime.utcnow().isoformat(),
                },
            )
    
    async def update_human_block(self, human: str) -> None:
        """Update the agent's human/user block."""
        self._agent_data["human_block"] = human[:self.config.human_block_limit]
        
        async with self.db.session(self.tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("UPDATE agents SET human_block = :human, updated_at = :now WHERE id = :id"),
                {
                    "id": str(self.agent_id),
                    "human": self._agent_data["human_block"],
                    "now": datetime.utcnow().isoformat(),
                },
            )
    
    async def get_memories(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get memories for the agent."""
        if user_id:
            memories = await self.memory.get_memories_for_user(
                agent_id=self.agent_id,
                user_id=user_id,
                tenant_id=self.tenant_id,
                limit=limit,
            )
        else:
            memories = await self.memory.memory_blocks.get_for_agent(
                agent_id=self.agent_id,
                tenant_id=self.tenant_id,
                limit=limit,
            )
        
        return [
            {
                "id": str(m.id),
                "type": m.block_type.value,
                "content": m.content,
                "importance": m.importance,
                "confidence": m.confidence,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ]
    
    async def close(self) -> None:
        """Close database connections."""
        await self.db.disconnect()
