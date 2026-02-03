"""Monthly integration: Deep personality and memory integration.

Implements monthly 'personality update' that integrates accumulated
experiences into core identity (persona block) and user understanding.
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from uuid import UUID
import json

from patina.database.connection import Database
from patina.database.models import JournalEntry, JournalEntryType, MemoryBlockType
from patina.memory.semantic import MemoryBlockStore


class MonthlyIntegrator:
    """Monthly deep integration of experiences into identity.
    
    Updates persona and human blocks based on accumulated patterns,
    merges redundant memories, and archives low-value data.
    """
    
    def __init__(self, db: Database, llm_client=None):
        self.db = db
        self.llm = llm_client
        self.memory_store = MemoryBlockStore(db)
    
    async def run(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        target_month: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Perform monthly integration.
        
        Args:
            agent_id: Agent to integrate
            tenant_id: Tenant context
            target_month: First day of target month
            
        Returns:
            Summary of integration operations
        """
        if target_month is None:
            today = datetime.utcnow().date()
            target_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        
        month_end = (target_month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        
        # Get agent current state
        agent = await self._get_agent(agent_id, tenant_id)
        if not agent:
            return {"error": "Agent not found"}
        
        # Get month's weekly syntheses
        syntheses = await self._get_weeks_syntheses(
            agent_id=agent_id,
            tenant_id=tenant_id,
            month_start=target_month,
            month_end=month_end,
        )
        
        # Get current memory blocks
        memories = await self.memory_store.get_for_agent(
            agent_id=agent_id,
            tenant_id=tenant_id,
            limit=500,
        )
        
        # Generate updated persona
        updated_persona = await self._generate_updated_persona(
            agent=agent,
            syntheses=syntheses,
            memories=memories,
        )
        
        # Generate updated human block
        updated_human = await self._generate_updated_human(
            agent=agent,
            syntheses=syntheses,
            memories=memories,
        )
        
        # Identify memory operations
        memory_ops = await self._plan_memory_operations(memories)
        
        # Execute updates
        await self._update_agent_blocks(
            agent_id=agent_id,
            tenant_id=tenant_id,
            persona=updated_persona,
            human=updated_human,
        )
        
        # Execute memory operations
        merged = await self._execute_merges(memory_ops.get("merges", []), tenant_id)
        archived = await self._execute_archives(memory_ops.get("archives", []), tenant_id)
        
        # Create integration journal
        entry = await self._create_integration_entry(
            agent_id=agent_id,
            tenant_id=tenant_id,
            persona_updated=updated_persona != agent.get("persona_block", ""),
            human_updated=updated_human != agent.get("human_block", ""),
            merged_count=merged,
            archived_count=archived,
            month=target_month,
        )
        
        return {
            "journal_entry_id": str(entry.id),
            "persona_updated": updated_persona != agent.get("persona_block", ""),
            "human_updated": updated_human != agent.get("human_block", ""),
            "memories_merged": merged,
            "memories_archived": archived,
        }
    
    async def _get_agent(self, agent_id: UUID, tenant_id: UUID) -> Optional[Dict]:
        """Get agent configuration."""
        row = await self.db.fetch_one(
            "SELECT * FROM agents WHERE id = :id",
            {"id": str(agent_id)},
            tenant_id,
        )
        return row
    
    async def _get_weeks_syntheses(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        month_start: date,
        month_end: date,
    ) -> List[Dict]:
        """Fetch weekly synthesis entries for the month."""
        rows = await self.db.fetch_all(
            """
            SELECT id, content, extracted_facts, emotional_valence,
                   significance_score, created_at
            FROM journal_entries
            WHERE agent_id = :agent_id
            AND entry_type = 'weekly_synthesis'
            AND DATE(created_at) >= :month_start
            AND DATE(created_at) <= :month_end
            ORDER BY created_at
            """,
            {
                "agent_id": str(agent_id),
                "month_start": month_start.isoformat(),
                "month_end": month_end.isoformat(),
            },
            tenant_id,
        )
        
        syntheses = []
        for row in rows:
            facts = row.get("extracted_facts", "[]")
            if isinstance(facts, str):
                facts = json.loads(facts)
            
            syntheses.append({
                "id": row["id"],
                "content": row["content"],
                "extracted_facts": facts,
                "emotional_valence": row.get("emotional_valence", 0),
                "significance": row.get("significance_score", 0.5),
            })
        
        return syntheses
    
    async def _generate_updated_persona(
        self,
        agent: Dict,
        syntheses: List[Dict],
        memories: List,
    ) -> str:
        """Generate updated persona block."""
        current = agent.get("persona_block", "")
        
        if self.llm:
            return await self._llm_update_persona(current, syntheses, memories)
        
        # No LLM - return current persona with timestamp
        return current
    
    async def _llm_update_persona(
        self,
        current_persona: str,
        syntheses: List[Dict],
        memories: List,
    ) -> str:
        """Update persona using LLM."""
        syntheses_text = "\n".join([
            f"Week: {s['content'][:200]}" for s in syntheses[:4]
        ])
        
        memories_text = "\n".join([
            f"- {m.content[:100]}" for m in memories[:20]
            if m.block_type in [MemoryBlockType.BELIEF, MemoryBlockType.PERSONALITY_TRAIT, MemoryBlockType.SKILL]
        ])
        
        prompt = f"""Update this AI assistant's persona based on the month's experiences.

Current persona:
{current_persona[:1000] if current_persona else "No current persona"}

This month's learnings:
{syntheses_text}

Relevant memory blocks:
{memories_text}

Write an updated persona (max 2000 chars) that:
1. Maintains core identity
2. Integrates demonstrated capabilities
3. Reflects behavioral patterns
4. Stays coherent and authentic

Only include the updated persona text, no explanations."""

        try:
            response = await self.llm.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            updated = response.choices[0].message.content.strip()
            return updated[:2000]
        except Exception:
            return current_persona
    
    async def _generate_updated_human(
        self,
        agent: Dict,
        syntheses: List[Dict],
        memories: List,
    ) -> str:
        """Generate updated human/user block."""
        current = agent.get("human_block", "")
        
        if self.llm:
            return await self._llm_update_human(current, syntheses, memories)
        
        # No LLM - aggregate facts from memories
        user_facts = [
            m.content for m in memories
            if m.block_type in [MemoryBlockType.FACT, MemoryBlockType.PREFERENCE]
            and m.user_id
        ][:10]
        
        if user_facts:
            return current + "\n\nRecent facts:\n" + "\n".join(f"- {f[:100]}" for f in user_facts)
        return current
    
    async def _llm_update_human(
        self,
        current_human: str,
        syntheses: List[Dict],
        memories: List,
    ) -> str:
        """Update human block using LLM."""
        user_facts = [
            m.content for m in memories
            if m.block_type in [MemoryBlockType.FACT, MemoryBlockType.PREFERENCE]
        ][:30]
        
        prompt = f"""Update the user profile based on accumulated knowledge.

Current user profile:
{current_human[:1000] if current_human else "No current profile"}

Facts about user(s):
{chr(10).join(f'- {f[:150]}' for f in user_facts)}

Write an updated user profile (max 2000 chars) that:
1. Consolidates redundant information
2. Prioritizes most important facts
3. Removes outdated information
4. Maintains respectful, accurate representation

Only include the updated profile text."""

        try:
            response = await self.llm.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            updated = response.choices[0].message.content.strip()
            return updated[:2000]
        except Exception:
            return current_human
    
    async def _plan_memory_operations(self, memories: List) -> Dict[str, List]:
        """Plan memory merge and archive operations."""
        merges = []
        archives = []
        
        # Group memories by type for potential merging
        by_type_user = {}
        for mem in memories:
            key = (mem.block_type, mem.user_id)
            if key not in by_type_user:
                by_type_user[key] = []
            by_type_user[key].append(mem)
        
        # Find similar memories to merge
        for key, mems in by_type_user.items():
            if len(mems) > 1:
                # Simple overlap detection
                for i, m1 in enumerate(mems):
                    for m2 in mems[i+1:]:
                        similarity = self._text_similarity(m1.content, m2.content)
                        if similarity > 0.7:
                            merges.append({
                                "source_ids": [m1.id, m2.id],
                                "merged_content": m1.content,  # Keep first
                            })
        
        # Archive low-importance, old memories
        cutoff = datetime.utcnow() - timedelta(days=90)
        for mem in memories:
            if mem.importance < 0.3 and mem.created_at < cutoff:
                if mem.id not in [m["source_ids"][1] for m in merges if len(m.get("source_ids", [])) > 1]:
                    archives.append(mem.id)
        
        return {"merges": merges, "archives": archives}
    
    def _text_similarity(self, a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    async def _update_agent_blocks(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        persona: str,
        human: str,
    ) -> None:
        """Update agent's persona and human blocks."""
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    UPDATE agents
                    SET persona_block = :persona,
                        human_block = :human,
                        updated_at = :now
                    WHERE id = :id
                """),
                {
                    "id": str(agent_id),
                    "persona": persona,
                    "human": human,
                    "now": datetime.utcnow().isoformat(),
                },
            )
    
    async def _execute_merges(
        self,
        merges: List[Dict],
        tenant_id: UUID,
    ) -> int:
        """Execute memory merges."""
        count = 0
        for merge in merges:
            source_ids = merge.get("source_ids", [])
            if len(source_ids) >= 2:
                # Keep first, mark others as superseded
                primary = source_ids[0]
                for secondary in source_ids[1:]:
                    await self.memory_store.supersede(
                        old_memory_id=secondary,
                        new_memory_id=primary,
                        tenant_id=tenant_id,
                    )
                    count += 1
        return count
    
    async def _execute_archives(
        self,
        archive_ids: List[UUID],
        tenant_id: UUID,
    ) -> int:
        """Execute memory archives."""
        for mem_id in archive_ids:
            async with self.db.session(tenant_id) as session:
                from sqlalchemy import text
                await session.execute(
                    text("""
                        UPDATE memory_blocks
                        SET metadata = json_set(COALESCE(metadata, '{}'), '$.archived', 1)
                        WHERE id = :id
                    """),
                    {"id": str(mem_id)},
                )
        return len(archive_ids)
    
    async def _create_integration_entry(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        persona_updated: bool,
        human_updated: bool,
        merged_count: int,
        archived_count: int,
        month: date,
    ) -> JournalEntry:
        """Create monthly integration journal entry."""
        from uuid import uuid4
        
        content = f"""Monthly Integration - {month.strftime('%B %Y')}

Persona block: {'Updated' if persona_updated else 'No changes'}
User profile: {'Updated' if human_updated else 'No changes'}
Memories merged: {merged_count}
Memories archived: {archived_count}

Integration complete."""

        entry = JournalEntry(
            id=uuid4(),
            tenant_id=tenant_id,
            agent_id=agent_id,
            entry_type=JournalEntryType.MONTHLY_INTEGRATION,
            content=content,
            significance_score=0.8,
            source_message_range={
                "month": month.isoformat(),
                "persona_updated": persona_updated,
                "human_updated": human_updated,
                "merged": merged_count,
                "archived": archived_count,
            },
        )
        
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    INSERT INTO journal_entries
                    (id, tenant_id, agent_id, entry_type, content,
                     significance_score, source_message_range, created_at)
                    VALUES (:id, :tenant_id, :agent_id, :entry_type, :content,
                            :significance_score, :source_range, :created_at)
                """),
                {
                    "id": str(entry.id),
                    "tenant_id": str(entry.tenant_id),
                    "agent_id": str(entry.agent_id),
                    "entry_type": entry.entry_type.value,
                    "content": entry.content,
                    "significance_score": entry.significance_score,
                    "source_range": json.dumps(entry.source_message_range),
                    "created_at": entry.created_at.isoformat(),
                },
            )
        
        return entry
