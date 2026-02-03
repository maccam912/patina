"""Semantic memory: Memory blocks and facts.

The neocortex-analog that accumulates patterns through repeated exposure,
supporting similarity-based retrieval via embeddings.
"""

from datetime import datetime
from typing import List, Optional, Dict
from uuid import UUID
import json

from patina.database.connection import Database
from patina.database.models import MemoryBlock, MemoryBlockType


class MemoryBlockStore:
    """Storage for semantic memory blocks."""
    
    def __init__(self, db: Database):
        self.db = db
    
    async def create(
        self,
        tenant_id: UUID,
        agent_id: UUID,
        block_type: MemoryBlockType,
        content: str,
        user_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        confidence: float = 0.8,
        source_journal_ids: Optional[List[UUID]] = None,
        metadata: Optional[Dict] = None,
        expires_at: Optional[datetime] = None,
    ) -> MemoryBlock:
        """Create a new memory block."""
        memory = MemoryBlock(
            tenant_id=tenant_id,
            agent_id=agent_id,
            user_id=user_id,
            block_type=block_type,
            content=content,
            embedding=embedding,
            importance=importance,
            confidence=confidence,
            source_journal_ids=source_journal_ids or [],
            metadata=metadata or {},
            expires_at=expires_at,
        )
        
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    INSERT INTO memory_blocks 
                    (id, tenant_id, agent_id, user_id, block_type, content, embedding,
                     stability, last_accessed_at, access_count, importance, confidence,
                     source_journal_ids, metadata, created_at, expires_at)
                    VALUES (:id, :tenant_id, :agent_id, :user_id, :block_type, :content, :embedding,
                            :stability, :last_accessed_at, :access_count, :importance, :confidence,
                            :source_journal_ids, :metadata, :created_at, :expires_at)
                """),
                {
                    "id": str(memory.id),
                    "tenant_id": str(memory.tenant_id),
                    "agent_id": str(memory.agent_id),
                    "user_id": memory.user_id,
                    "block_type": memory.block_type.value,
                    "content": memory.content,
                    "embedding": json.dumps(memory.embedding) if memory.embedding else None,
                    "stability": memory.stability,
                    "last_accessed_at": memory.last_accessed_at.isoformat(),
                    "access_count": memory.access_count,
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                    "source_journal_ids": json.dumps([str(x) for x in memory.source_journal_ids]),
                    "metadata": json.dumps(memory.metadata),
                    "created_at": memory.created_at.isoformat(),
                    "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
                }
            )
        
        return memory
    
    async def get(self, memory_id: UUID, tenant_id: UUID) -> Optional[MemoryBlock]:
        """Get memory block by ID."""
        row = await self.db.fetch_one(
            "SELECT * FROM memory_blocks WHERE id = :id",
            {"id": str(memory_id)},
            tenant_id,
        )
        if row:
            return self._row_to_memory(row)
        return None
    
    async def get_for_agent(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        block_types: Optional[List[MemoryBlockType]] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        min_importance: float = 0.0,
        include_superseded: bool = False,
    ) -> List[MemoryBlock]:
        """Get memory blocks for an agent with optional filtering."""
        query = """
            SELECT * FROM memory_blocks 
            WHERE agent_id = :agent_id
            AND importance >= :min_importance
        """
        params: Dict = {
            "agent_id": str(agent_id),
            "min_importance": min_importance,
        }
        
        if not include_superseded:
            query += " AND superseded_by IS NULL"
        
        if user_id:
            query += " AND (user_id = :user_id OR user_id IS NULL)"
            params["user_id"] = user_id
        
        if block_types:
            type_list = ",".join(f"'{t.value}'" for t in block_types)
            query += f" AND block_type IN ({type_list})"
        
        query += " ORDER BY importance DESC, created_at DESC LIMIT :limit"
        params["limit"] = limit
        
        rows = await self.db.fetch_all(query, params, tenant_id)
        return [self._row_to_memory(r) for r in rows]
    
    async def get_for_user(
        self,
        agent_id: UUID,
        user_id: str,
        tenant_id: UUID,
        block_types: Optional[List[MemoryBlockType]] = None,
        limit: int = 50,
    ) -> List[MemoryBlock]:
        """Get memory blocks for a specific user."""
        query = """
            SELECT * FROM memory_blocks 
            WHERE agent_id = :agent_id AND user_id = :user_id
            AND superseded_by IS NULL
        """
        params: Dict = {
            "agent_id": str(agent_id),
            "user_id": user_id,
        }
        
        if block_types:
            type_list = ",".join(f"'{t.value}'" for t in block_types)
            query += f" AND block_type IN ({type_list})"
        
        query += " ORDER BY importance DESC LIMIT :limit"
        params["limit"] = limit
        
        rows = await self.db.fetch_all(query, params, tenant_id)
        return [self._row_to_memory(r) for r in rows]
    
    async def update_access(
        self,
        memory_id: UUID,
        tenant_id: UUID,
        stability_multiplier: float = 2.5,
    ) -> None:
        """Update access tracking (spaced repetition effect).
        
        Increases stability based on successful retrieval,
        implementing spaced repetition principles.
        """
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    UPDATE memory_blocks 
                    SET access_count = access_count + 1,
                        last_accessed_at = :now,
                        stability = stability * :multiplier
                    WHERE id = :id
                """),
                {
                    "id": str(memory_id),
                    "now": datetime.utcnow().isoformat(),
                    "multiplier": stability_multiplier,
                }
            )
    
    async def update_confidence(
        self,
        memory_id: UUID,
        confidence_delta: float,
        tenant_id: UUID,
    ) -> None:
        """Adjust confidence level for a memory."""
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    UPDATE memory_blocks 
                    SET confidence = MIN(MAX(confidence + :delta, 0), 1.0)
                    WHERE id = :id
                """),
                {"id": str(memory_id), "delta": confidence_delta},
            )
    
    async def supersede(
        self,
        old_memory_id: UUID,
        new_memory_id: UUID,
        tenant_id: UUID,
    ) -> None:
        """Mark a memory as superseded by a newer version."""
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("UPDATE memory_blocks SET superseded_by = :new_id WHERE id = :old_id"),
                {"old_id": str(old_memory_id), "new_id": str(new_memory_id)},
            )
    
    async def update_embedding(
        self,
        memory_id: UUID,
        embedding: List[float],
        tenant_id: UUID,
    ) -> None:
        """Store embedding for a memory block."""
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("UPDATE memory_blocks SET embedding = :embedding WHERE id = :id"),
                {"id": str(memory_id), "embedding": json.dumps(embedding)},
            )
    
    async def find_similar(
        self,
        agent_id: UUID,
        embedding: List[float],
        tenant_id: UUID,
        threshold: float = 0.85,
        limit: int = 10,
    ) -> List[MemoryBlock]:
        """Find similar memory blocks by embedding.
        
        Note: For SQLite, this uses a simple cosine similarity calculation.
        For PostgreSQL with pgvector, this would use the <=> operator.
        """
        # For SQLite, we fetch all and compute similarity in Python
        # In production PostgreSQL, this would be an index query
        all_memories = await self.get_for_agent(
            agent_id=agent_id,
            tenant_id=tenant_id,
            limit=1000,
        )
        
        # Filter by those with embeddings and compute similarity
        similar = []
        for mem in all_memories:
            if mem.embedding:
                sim = self._cosine_similarity(embedding, mem.embedding)
                if sim >= threshold:
                    similar.append((sim, mem))
        
        # Sort by similarity and return top matches
        similar.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in similar[:limit]]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def delete(self, memory_id: UUID, tenant_id: UUID) -> None:
        """Delete a memory block (hard delete)."""
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("DELETE FROM memory_blocks WHERE id = :id"),
                {"id": str(memory_id)},
            )
    
    async def get_for_decay_check(
        self,
        tenant_id: UUID,
        limit: int = 1000,
    ) -> List[MemoryBlock]:
        """Get memories for decay cycle processing."""
        rows = await self.db.fetch_all(
            """
            SELECT * FROM memory_blocks 
            WHERE tenant_id = :tenant_id AND superseded_by IS NULL
            ORDER BY last_accessed_at ASC LIMIT :limit
            """,
            {"tenant_id": str(tenant_id), "limit": limit},
            tenant_id,
        )
        return [self._row_to_memory(r) for r in rows]
    
    def _row_to_memory(self, row: Dict) -> MemoryBlock:
        """Convert database row to MemoryBlock model."""
        embedding = row.get("embedding")
        if isinstance(embedding, str) and embedding:
            embedding = json.loads(embedding)
        
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        source_ids = row.get("source_journal_ids", "[]")
        if isinstance(source_ids, str):
            source_ids = json.loads(source_ids)
        source_ids = [UUID(x) if isinstance(x, str) else x for x in source_ids]
        
        def parse_dt(val):
            if val is None:
                return None
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return val
        
        return MemoryBlock(
            id=UUID(row["id"]),
            tenant_id=UUID(row["tenant_id"]),
            agent_id=UUID(row["agent_id"]),
            user_id=row.get("user_id"),
            block_type=MemoryBlockType(row["block_type"]),
            content=row["content"],
            embedding=embedding,
            stability=row.get("stability", 1.0),
            last_accessed_at=parse_dt(row.get("last_accessed_at")) or datetime.utcnow(),
            access_count=row.get("access_count", 1),
            next_review_at=parse_dt(row.get("next_review_at")),
            importance=row.get("importance", 0.5),
            confidence=row.get("confidence", 0.8),
            source_journal_ids=source_ids,
            superseded_by=UUID(row["superseded_by"]) if row.get("superseded_by") else None,
            metadata=metadata,
            created_at=parse_dt(row["created_at"]) or datetime.utcnow(),
            expires_at=parse_dt(row.get("expires_at")),
        )
