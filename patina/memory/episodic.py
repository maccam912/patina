"""Episodic memory: Conversations and messages.

The hippocampus-analog that captures complete interaction records
with temporal/relational context.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
import json

from patina.database.connection import Database
from patina.database.models import Conversation, Message, MessageRole


class ConversationStore:
    """Storage for conversation sessions."""
    
    def __init__(self, db: Database):
        self.db = db
    
    async def create(
        self,
        tenant_id: UUID,
        agent_id: UUID,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            tenant_id=tenant_id,
            agent_id=agent_id,
            user_id=user_id,
            title=title,
            metadata=metadata or {},
        )
        
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    INSERT INTO conversations 
                    (id, tenant_id, agent_id, user_id, title, metadata, created_at, updated_at)
                    VALUES (:id, :tenant_id, :agent_id, :user_id, :title, :metadata, :created_at, :updated_at)
                """),
                {
                    "id": str(conversation.id),
                    "tenant_id": str(conversation.tenant_id),
                    "agent_id": str(conversation.agent_id),
                    "user_id": conversation.user_id,
                    "title": conversation.title,
                    "metadata": json.dumps(conversation.metadata),
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat(),
                }
            )
        
        return conversation
    
    async def get(self, conversation_id: UUID, tenant_id: UUID) -> Optional[Conversation]:
        """Get conversation by ID."""
        row = await self.db.fetch_one(
            "SELECT * FROM conversations WHERE id = :id",
            {"id": str(conversation_id)},
            tenant_id,
        )
        if row:
            return self._row_to_conversation(row)
        return None
    
    async def list_for_agent(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
    ) -> List[Conversation]:
        """List conversations for an agent."""
        query = """
            SELECT * FROM conversations 
            WHERE agent_id = :agent_id
        """
        if not include_archived:
            query += " AND is_archived = 0"
        query += " ORDER BY updated_at DESC LIMIT :limit OFFSET :offset"
        
        rows = await self.db.fetch_all(
            query,
            {"agent_id": str(agent_id), "limit": limit, "offset": offset},
            tenant_id,
        )
        return [self._row_to_conversation(r) for r in rows]
    
    async def list_for_user(
        self,
        user_id: str,
        agent_id: UUID,
        tenant_id: UUID,
        limit: int = 50,
    ) -> List[Conversation]:
        """List conversations for a specific user."""
        rows = await self.db.fetch_all(
            """
            SELECT * FROM conversations 
            WHERE user_id = :user_id AND agent_id = :agent_id
            ORDER BY updated_at DESC LIMIT :limit
            """,
            {"user_id": user_id, "agent_id": str(agent_id), "limit": limit},
            tenant_id,
        )
        return [self._row_to_conversation(r) for r in rows]
    
    async def update_summary(
        self,
        conversation_id: UUID,
        summary: str,
        summary_token_count: int,
        tenant_id: UUID,
    ) -> None:
        """Update rolling summary for a conversation."""
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    UPDATE conversations 
                    SET summary = :summary, 
                        summary_token_count = :count,
                        updated_at = :updated_at
                    WHERE id = :id
                """),
                {
                    "id": str(conversation_id),
                    "summary": summary,
                    "count": summary_token_count,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )
    
    async def increment_message_count(
        self,
        conversation_id: UUID,
        tenant_id: UUID,
    ) -> None:
        """Increment message count for conversation."""
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    UPDATE conversations 
                    SET message_count = message_count + 1,
                        updated_at = :updated_at
                    WHERE id = :id
                """),
                {
                    "id": str(conversation_id),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )
    
    async def archive(self, conversation_id: UUID, tenant_id: UUID) -> None:
        """Archive a conversation."""
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("UPDATE conversations SET is_archived = 1 WHERE id = :id"),
                {"id": str(conversation_id)},
            )
    
    def _row_to_conversation(self, row: Dict) -> Conversation:
        """Convert database row to Conversation model."""
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        return Conversation(
            id=UUID(row["id"]),
            tenant_id=UUID(row["tenant_id"]),
            agent_id=UUID(row["agent_id"]),
            user_id=row.get("user_id"),
            title=row.get("title"),
            summary=row.get("summary"),
            summary_token_count=row.get("summary_token_count", 0),
            message_count=row.get("message_count", 0),
            metadata=metadata,
            is_archived=bool(row.get("is_archived", 0)),
            created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
            updated_at=datetime.fromisoformat(row["updated_at"]) if isinstance(row["updated_at"], str) else row["updated_at"],
        )


class MessageStore:
    """Storage for individual messages."""
    
    def __init__(self, db: Database):
        self.db = db
    
    async def create(
        self,
        conversation_id: UUID,
        tenant_id: UUID,
        role: MessageRole,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        tool_call_id: Optional[str] = None,
        token_count: Optional[int] = None,
        importance_score: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> Message:
        """Create a new message."""
        message = Message(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            token_count=token_count,
            importance_score=importance_score,
            metadata=metadata or {},
        )
        
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    INSERT INTO messages 
                    (id, conversation_id, tenant_id, role, content, tool_calls, 
                     tool_call_id, token_count, importance_score, metadata, created_at)
                    VALUES (:id, :conversation_id, :tenant_id, :role, :content, :tool_calls,
                            :tool_call_id, :token_count, :importance_score, :metadata, :created_at)
                """),
                {
                    "id": str(message.id),
                    "conversation_id": str(message.conversation_id),
                    "tenant_id": str(message.tenant_id),
                    "role": message.role.value,
                    "content": message.content,
                    "tool_calls": json.dumps(message.tool_calls) if message.tool_calls else None,
                    "tool_call_id": message.tool_call_id,
                    "token_count": message.token_count,
                    "importance_score": message.importance_score,
                    "metadata": json.dumps(message.metadata),
                    "created_at": message.created_at.isoformat(),
                }
            )
        
        return message
    
    async def get_for_conversation(
        self,
        conversation_id: UUID,
        tenant_id: UUID,
        limit: int = 100,
        before: Optional[datetime] = None,
    ) -> List[Message]:
        """Get messages for a conversation."""
        query = """
            SELECT * FROM messages 
            WHERE conversation_id = :conversation_id
        """
        params: Dict[str, Any] = {"conversation_id": str(conversation_id)}
        
        if before:
            query += " AND created_at < :before"
            params["before"] = before.isoformat()
        
        query += " ORDER BY created_at ASC LIMIT :limit"
        params["limit"] = limit
        
        rows = await self.db.fetch_all(query, params, tenant_id)
        return [self._row_to_message(r) for r in rows]
    
    async def get_recent_for_agent(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Message]:
        """Get recent messages across all conversations for an agent."""
        query = """
            SELECT m.* FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.agent_id = :agent_id
        """
        params: Dict[str, Any] = {"agent_id": str(agent_id)}
        
        if since:
            query += " AND m.created_at >= :since"
            params["since"] = since.isoformat()
        
        query += " ORDER BY m.created_at DESC LIMIT :limit"
        params["limit"] = limit
        
        rows = await self.db.fetch_all(query, params, tenant_id)
        return [self._row_to_message(r) for r in rows]
    
    async def mark_summarized(
        self,
        message_ids: List[UUID],
        tenant_id: UUID,
    ) -> None:
        """Mark messages as having been summarized."""
        if not message_ids:
            return
        
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            # SQLite doesn't support array syntax, so we do it one by one
            for msg_id in message_ids:
                await session.execute(
                    text("UPDATE messages SET is_summarized = 1 WHERE id = :id"),
                    {"id": str(msg_id)},
                )
    
    async def update_importance(
        self,
        message_id: UUID,
        importance_score: float,
        tenant_id: UUID,
    ) -> None:
        """Update importance score for a message."""
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("UPDATE messages SET importance_score = :score WHERE id = :id"),
                {"id": str(message_id), "score": importance_score},
            )
    
    async def update_embedding(
        self,
        message_id: UUID,
        embedding: List[float],
        tenant_id: UUID,
    ) -> None:
        """Store embedding for a message."""
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("UPDATE messages SET embedding = :embedding WHERE id = :id"),
                {"id": str(message_id), "embedding": json.dumps(embedding)},
            )
    
    def _row_to_message(self, row: Dict) -> Message:
        """Convert database row to Message model."""
        tool_calls = row.get("tool_calls")
        if isinstance(tool_calls, str):
            tool_calls = json.loads(tool_calls) if tool_calls else None
        
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        embedding = row.get("embedding")
        if isinstance(embedding, str) and embedding:
            embedding = json.loads(embedding)
        
        return Message(
            id=UUID(row["id"]),
            conversation_id=UUID(row["conversation_id"]),
            tenant_id=UUID(row["tenant_id"]),
            role=MessageRole(row["role"]),
            content=row["content"],
            tool_calls=tool_calls,
            tool_call_id=row.get("tool_call_id"),
            embedding=embedding,
            token_count=row.get("token_count"),
            importance_score=row.get("importance_score", 0.5),
            is_summarized=bool(row.get("is_summarized", 0)),
            metadata=metadata,
            created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
        )
    
    def calculate_importance(
        self,
        content: str,
        role: MessageRole,
        has_tool_calls: bool = False,
    ) -> float:
        """Calculate importance score for a message.
        
        Based on:
        - Emotional content indicators
        - Decision markers
        - Novel information signals
        - Tool usage
        """
        score = 0.5  # Base score
        
        # Emotional indicators
        emotional_markers = [
            "love", "hate", "angry", "happy", "sad", "excited",
            "worried", "concerned", "grateful", "frustrated",
            "!", "?!",
        ]
        content_lower = content.lower()
        for marker in emotional_markers:
            if marker in content_lower:
                score += 0.05
        
        # Decision markers
        decision_markers = [
            "decide", "choose", "commit", "will do", "promise",
            "agree", "confirm", "yes", "no", "definitely",
        ]
        for marker in decision_markers:
            if marker in content_lower:
                score += 0.1
        
        # Personal information
        personal_markers = [
            "my name", "i am", "i'm", "my job", "my family",
            "my wife", "my husband", "my kids", "i work",
        ]
        for marker in personal_markers:
            if marker in content_lower:
                score += 0.15
        
        # Tool calls are important
        if has_tool_calls:
            score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
