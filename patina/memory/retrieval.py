"""Memory retrieval: Hybrid search across memory stores.

Combines vector similarity, full-text search, and trigram matching
for optimal memory retrieval.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
import math

from patina.database.connection import Database
from patina.database.models import MemoryBlock, MemoryBlockType


class MemoryRetriever:
    """Hybrid memory retrieval with multiple search strategies."""
    
    def __init__(self, db: Database, embedding_fn=None):
        self.db = db
        self._embedding_fn = embedding_fn
    
    async def search(
        self,
        query: str,
        agent_id: UUID,
        tenant_id: UUID,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryBlockType]] = None,
        limit: int = 10,
        use_embedding: bool = True,
        use_text: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search memories using hybrid approach.
        
        Combines:
        1. Vector similarity (if embeddings available)
        2. Full-text search
        3. Keyword/trigram matching
        
        Returns results with relevance scores.
        """
        results = []
        
        # Text-based search
        if use_text:
            text_results = await self._text_search(
                query=query,
                agent_id=agent_id,
                tenant_id=tenant_id,
                user_id=user_id,
                memory_types=memory_types,
                limit=limit * 2,  # Get more for deduplication
            )
            results.extend(text_results)
        
        # Vector similarity search
        if use_embedding and self._embedding_fn:
            try:
                query_embedding = await self._embedding_fn(query)
                embedding_results = await self._embedding_search(
                    embedding=query_embedding,
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    memory_types=memory_types,
                    limit=limit * 2,
                )
                results.extend(embedding_results)
            except Exception:
                pass  # Fallback to text-only if embedding fails
        
        # Deduplicate and rank
        ranked = self._rank_results(results, limit)
        
        # Update access for retrieved memories
        for result in ranked:
            await self._update_access(result["memory"].id, tenant_id)
        
        return ranked
    
    async def _text_search(
        self,
        query: str,
        agent_id: UUID,
        tenant_id: UUID,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryBlockType]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Full-text and keyword search."""
        # Simple LIKE-based search for SQLite compatibility
        # PostgreSQL would use ts_rank and to_tsquery
        search_term = f"%{query.lower()}%"
        
        sql = """
            SELECT * FROM memory_blocks 
            WHERE agent_id = :agent_id
            AND LOWER(content) LIKE :search_term
            AND superseded_by IS NULL
        """
        params: Dict[str, Any] = {
            "agent_id": str(agent_id),
            "search_term": search_term,
        }
        
        if user_id:
            sql += " AND (user_id = :user_id OR user_id IS NULL)"
            params["user_id"] = user_id
        
        if memory_types:
            type_list = ",".join(f"'{t.value}'" for t in memory_types)
            sql += f" AND block_type IN ({type_list})"
        
        sql += " ORDER BY importance DESC LIMIT :limit"
        params["limit"] = limit
        
        rows = await self.db.fetch_all(sql, params, tenant_id)
        
        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            # Simple relevance based on keyword match frequency
            content_lower = memory.content.lower()
            query_words = query.lower().split()
            match_count = sum(1 for word in query_words if word in content_lower)
            relevance = match_count / max(len(query_words), 1)
            
            results.append({
                "memory": memory,
                "score": relevance * 0.5 + memory.importance * 0.3 + memory.confidence * 0.2,
                "source": "text",
            })
        
        return results
    
    async def _embedding_search(
        self,
        embedding: List[float],
        agent_id: UUID,
        tenant_id: UUID,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryBlockType]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search.
        
        Note: For production PostgreSQL with pgvector, this would use:
        ORDER BY embedding <=> :query_embedding
        """
        # Fetch all memories with embeddings (SQLite fallback)
        sql = """
            SELECT * FROM memory_blocks 
            WHERE agent_id = :agent_id
            AND embedding IS NOT NULL
            AND superseded_by IS NULL
        """
        params: Dict[str, Any] = {"agent_id": str(agent_id)}
        
        if user_id:
            sql += " AND (user_id = :user_id OR user_id IS NULL)"
            params["user_id"] = user_id
        
        if memory_types:
            type_list = ",".join(f"'{t.value}'" for t in memory_types)
            sql += f" AND block_type IN ({type_list})"
        
        rows = await self.db.fetch_all(sql, params, tenant_id)
        
        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            if memory.embedding:
                similarity = self._cosine_similarity(embedding, memory.embedding)
                # Weight by importance and confidence
                score = similarity * 0.6 + memory.importance * 0.25 + memory.confidence * 0.15
                
                results.append({
                    "memory": memory,
                    "score": score,
                    "similarity": similarity,
                    "source": "embedding",
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def _rank_results(
        self,
        results: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Deduplicate and rank combined results."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            memory_id = result["memory"].id
            if memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_results.append(result)
        
        # Sort by combined score
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        return unique_results[:limit]
    
    async def _update_access(self, memory_id: UUID, tenant_id: UUID) -> None:
        """Update access tracking for retrieved memory."""
        from patina.memory.semantic import MemoryBlockStore
        store = MemoryBlockStore(self.db)
        await store.update_access(memory_id, tenant_id)
    
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
    
    def calculate_retrievability(
        self,
        last_accessed: datetime,
        stability: float,
        now: Optional[datetime] = None,
    ) -> float:
        """Calculate current memory retrievability using Ebbinghaus curve.
        
        R = e^(-t/S)
        
        Where:
        - R = retrievability (probability of successful recall)
        - t = time since last access (in days)
        - S = stability (increases with each retrieval)
        """
        now = now or datetime.utcnow()
        days_elapsed = (now - last_accessed).total_seconds() / 86400
        return math.exp(-days_elapsed / stability)
    
    async def get_by_retrievability(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        min_retrievability: float = 0.1,
        max_retrievability: float = 1.0,
    ) -> List[MemoryBlock]:
        """Get memories filtered by calculated retrievability."""
        from patina.memory.semantic import MemoryBlockStore
        store = MemoryBlockStore(self.db)
        
        memories = await store.get_for_agent(
            agent_id=agent_id,
            tenant_id=tenant_id,
            limit=1000,
        )
        
        now = datetime.utcnow()
        filtered = []
        
        for mem in memories:
            r = self.calculate_retrievability(
                mem.last_accessed_at,
                mem.stability,
                now,
            )
            if min_retrievability <= r <= max_retrievability:
                filtered.append(mem)
        
        return filtered
    
    def _row_to_memory(self, row: Dict) -> MemoryBlock:
        """Convert database row to MemoryBlock model."""
        import json
        
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
