"""Memory decay: Ebbinghaus forgetting curve implementation.

Implements R = e^(-t/S) where:
- R = retrievability
- t = time since last access
- S = stability (increases with retrieval)
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID
import math

from patina.database.connection import Database
from patina.database.models import MemoryBlock
from patina.memory.semantic import MemoryBlockStore


class MemoryDecayManager:
    """Manages memory decay using Ebbinghaus forgetting curve.
    
    Memories decay over time unless reinforced through access.
    This implements the biological principle of "use it or lose it"
    for memory systems.
    """
    
    def __init__(
        self,
        db: Database,
        archive_threshold: float = 0.1,
        prune_threshold: float = 0.05,
        importance_prune_threshold: float = 0.3,
    ):
        self.db = db
        self.archive_threshold = archive_threshold  # Archive when R < this
        self.prune_threshold = prune_threshold      # Delete when R < this
        self.importance_prune_threshold = importance_prune_threshold
        self.store = MemoryBlockStore(db)
    
    def calculate_retrievability(
        self,
        last_accessed: datetime,
        stability: float,
        now: Optional[datetime] = None,
    ) -> float:
        """Calculate current memory retrievability.
        
        R = e^(-t/S)
        
        Args:
            last_accessed: When memory was last accessed
            stability: Memory stability (increases with each access)
            now: Current time (defaults to UTC now)
            
        Returns:
            Retrievability between 0 and 1
        """
        now = now or datetime.utcnow()
        days_elapsed = (now - last_accessed).total_seconds() / 86400
        return math.exp(-days_elapsed / stability)
    
    def calculate_next_review(
        self,
        stability: float,
        target_retrievability: float = 0.9,
    ) -> float:
        """Calculate days until next review to maintain target retrievability.
        
        From R = e^(-t/S), solving for t:
        t = -S * ln(R)
        """
        return -stability * math.log(target_retrievability)
    
    async def run_decay_cycle(
        self,
        tenant_id: UUID,
        agent_id: Optional[UUID] = None,
    ) -> Dict[str, int]:
        """Apply forgetting curve to all memories.
        
        Returns counts of archived and pruned memories.
        """
        memories = await self.store.get_for_decay_check(tenant_id)
        
        if agent_id:
            memories = [m for m in memories if m.agent_id == agent_id]
        
        to_archive = []
        to_prune = []
        now = datetime.utcnow()
        
        for mem in memories:
            R = self.calculate_retrievability(
                mem.last_accessed_at,
                mem.stability,
                now,
            )
            
            # Very low retrievability + low importance = prune
            if R < self.prune_threshold and mem.importance < self.importance_prune_threshold:
                to_prune.append(mem.id)
            # Low retrievability = archive
            elif R < self.archive_threshold:
                to_archive.append(mem.id)
        
        # Archive memories (soft delete)
        if to_archive:
            await self._archive_memories(to_archive, tenant_id)
        
        # Prune memories (hard delete)
        if to_prune:
            await self._prune_memories(to_prune, tenant_id)
        
        return {
            "archived": len(to_archive),
            "pruned": len(to_prune),
            "total_checked": len(memories),
        }
    
    async def _archive_memories(
        self,
        memory_ids: List[UUID],
        tenant_id: UUID,
    ) -> None:
        """Archive low-retrievability memories.
        
        In a full implementation, this would move to an archive table.
        For now, we mark them as superseded by None (effectively archived).
        """
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            for mem_id in memory_ids:
                # Create a "archived" marker in metadata
                await session.execute(
                    text("""
                        UPDATE memory_blocks 
                        SET metadata = json_set(COALESCE(metadata, '{}'), '$.archived', 1),
                            metadata = json_set(COALESCE(metadata, '{}'), '$.archived_at', :now)
                        WHERE id = :id
                    """),
                    {"id": str(mem_id), "now": datetime.utcnow().isoformat()},
                )
    
    async def _prune_memories(
        self,
        memory_ids: List[UUID],
        tenant_id: UUID,
    ) -> None:
        """Hard delete memories that have decayed beyond recovery."""
        for mem_id in memory_ids:
            await self.store.delete(mem_id, tenant_id)
    
    def get_decay_stats(
        self,
        memories: List[MemoryBlock],
        now: Optional[datetime] = None,
    ) -> Dict:
        """Get statistics about memory decay state."""
        now = now or datetime.utcnow()
        
        retrievabilities = [
            self.calculate_retrievability(m.last_accessed_at, m.stability, now)
            for m in memories
        ]
        
        if not retrievabilities:
            return {
                "count": 0,
                "avg_retrievability": 0,
                "min_retrievability": 0,
                "max_retrievability": 0,
                "at_risk_count": 0,
                "healthy_count": 0,
            }
        
        at_risk = sum(1 for r in retrievabilities if r < self.archive_threshold)
        healthy = sum(1 for r in retrievabilities if r >= 0.5)
        
        return {
            "count": len(memories),
            "avg_retrievability": sum(retrievabilities) / len(retrievabilities),
            "min_retrievability": min(retrievabilities),
            "max_retrievability": max(retrievabilities),
            "at_risk_count": at_risk,
            "healthy_count": healthy,
        }
