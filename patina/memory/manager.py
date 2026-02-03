"""Memory manager: Unified interface for all memory operations."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from patina.config import PatinaConfig
from patina.database.connection import Database
from patina.database.models import (
    Conversation,
    Message,
    MessageRole,
    MemoryBlock,
    MemoryBlockType,
)
from patina.memory.working import ContextWindowManager, ContextBudget
from patina.memory.episodic import ConversationStore, MessageStore
from patina.memory.semantic import MemoryBlockStore
from patina.memory.retrieval import MemoryRetriever


class MemoryManager:
    """Unified interface for memory operations.
    
    Provides high-level methods that coordinate across working,
    episodic, and semantic memory stores.
    """
    
    def __init__(
        self,
        db: Database,
        config: Optional[PatinaConfig] = None,
        embedding_fn=None,
        llm_client=None,
    ):
        self.db = db
        self.config = config or PatinaConfig()
        
        # Initialize stores
        self.conversations = ConversationStore(db)
        self.messages = MessageStore(db)
        self.memory_blocks = MemoryBlockStore(db)
        self.retriever = MemoryRetriever(db, embedding_fn)
        
        # Context window manager
        budget = ContextBudget(
            total_tokens=self.config.context_total_tokens,
            output_reserved=self.config.context_output_reserved,
        )
        self.context = ContextWindowManager(budget, llm_client)
        
        self._embedding_fn = embedding_fn
    
    # =========================================================================
    # CONVERSATION OPERATIONS
    # =========================================================================
    
    async def start_conversation(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Conversation:
        """Start a new conversation session."""
        return await self.conversations.create(
            tenant_id=tenant_id,
            agent_id=agent_id,
            user_id=user_id,
            title=title,
        )
    
    async def get_conversation(
        self,
        conversation_id: UUID,
        tenant_id: UUID,
    ) -> Optional[Conversation]:
        """Get conversation by ID."""
        return await self.conversations.get(conversation_id, tenant_id)
    
    async def list_conversations(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Conversation]:
        """List conversations for an agent, optionally filtered by user."""
        if user_id:
            return await self.conversations.list_for_user(
                user_id=user_id,
                agent_id=agent_id,
                tenant_id=tenant_id,
                limit=limit,
            )
        return await self.conversations.list_for_agent(
            agent_id=agent_id,
            tenant_id=tenant_id,
            limit=limit,
        )
    
    # =========================================================================
    # MESSAGE OPERATIONS
    # =========================================================================
    
    async def add_message(
        self,
        conversation_id: UUID,
        tenant_id: UUID,
        role: MessageRole,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Message:
        """Add a message to a conversation."""
        # Calculate importance
        importance = self.messages.calculate_importance(
            content=content,
            role=role,
            has_tool_calls=bool(tool_calls),
        )
        
        # Count tokens
        token_count = self.context.count_tokens(content)
        
        # Create message
        message = await self.messages.create(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            token_count=token_count,
            importance_score=importance,
            metadata=metadata,
        )
        
        # Update conversation stats
        await self.conversations.increment_message_count(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
        )
        
        # Generate embedding async (non-blocking)
        if self._embedding_fn:
            try:
                embedding = await self._embedding_fn(content)
                await self.messages.update_embedding(
                    message_id=message.id,
                    embedding=embedding,
                    tenant_id=tenant_id,
                )
            except Exception:
                pass  # Embedding failure shouldn't block message creation
        
        return message
    
    async def get_messages(
        self,
        conversation_id: UUID,
        tenant_id: UUID,
        limit: int = 100,
    ) -> List[Message]:
        """Get messages for a conversation."""
        return await self.messages.get_for_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            limit=limit,
        )
    
    # =========================================================================
    # MEMORY BLOCK OPERATIONS
    # =========================================================================
    
    async def store_memory(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        content: str,
        block_type: MemoryBlockType,
        user_id: Optional[str] = None,
        importance: float = 0.5,
        confidence: float = 0.8,
        metadata: Optional[Dict] = None,
    ) -> MemoryBlock:
        """Store a new memory block."""
        # Generate embedding
        embedding = None
        if self._embedding_fn:
            try:
                embedding = await self._embedding_fn(content)
            except Exception:
                pass
        
        return await self.memory_blocks.create(
            tenant_id=tenant_id,
            agent_id=agent_id,
            user_id=user_id,
            block_type=block_type,
            content=content,
            embedding=embedding,
            importance=importance,
            confidence=confidence,
            metadata=metadata,
        )
    
    async def search_memories(
        self,
        query: str,
        agent_id: UUID,
        tenant_id: UUID,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryBlockType]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search memories using hybrid retrieval."""
        return await self.retriever.search(
            query=query,
            agent_id=agent_id,
            tenant_id=tenant_id,
            user_id=user_id,
            memory_types=memory_types,
            limit=limit,
        )
    
    async def get_memories_for_user(
        self,
        agent_id: UUID,
        user_id: str,
        tenant_id: UUID,
        limit: int = 50,
    ) -> List[MemoryBlock]:
        """Get all memories for a specific user."""
        return await self.memory_blocks.get_for_user(
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            limit=limit,
        )
    
    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================
    
    async def build_context(
        self,
        agent: Dict,
        conversation_id: UUID,
        tenant_id: UUID,
        current_query: str,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """Build complete context for LLM call.
        
        Assembles:
        1. System prompt with persona
        2. Retrieved relevant memories
        3. Conversation history (compressed if needed)
        """
        # Get conversation messages
        messages = await self.messages.get_for_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
        )
        
        # Convert to dict format
        message_dicts = [
            {"role": m.role.value, "content": m.content}
            for m in messages
        ]
        
        # Search for relevant memories
        memory_results = await self.search_memories(
            query=current_query,
            agent_id=agent["id"],
            tenant_id=tenant_id,
            user_id=user_id,
            limit=10,
        )
        
        # Extract memory content
        retrieved_memories = [
            {"content": r["memory"].content, "score": r["score"]}
            for r in memory_results
        ]
        
        # Build context with working memory manager
        return await self.context.build_context(
            agent=agent,
            messages=message_dicts,
            retrieved_memories=retrieved_memories,
            current_query=current_query,
        )
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return self.context.count_tokens(text)
