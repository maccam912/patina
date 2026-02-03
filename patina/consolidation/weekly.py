"""Weekly synthesis: Pattern extraction from daily journals.

Implements weekly analysis that identifies recurring themes and
promotes validated facts to semantic memory.
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from uuid import UUID
import json

from patina.database.connection import Database
from patina.database.models import JournalEntry, JournalEntryType, MemoryBlockType
from patina.memory.semantic import MemoryBlockStore


class WeeklySynthesizer:
    """Weekly synthesis from daily journal entries.
    
    Identifies patterns across the week, promotes recurring facts
    to semantic memory, and flags contradictions.
    """
    
    def __init__(self, db: Database, llm_client=None):
        self.db = db
        self.llm = llm_client
        self.memory_store = MemoryBlockStore(db)
    
    async def run(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        week_start: Optional[date] = None,
        user_id: Optional[str] = None,
    ) -> Optional[JournalEntry]:
        """Generate weekly synthesis from daily journals.
        
        Args:
            agent_id: Agent to synthesize for
            tenant_id: Tenant context
            week_start: Start of week (defaults to last Sunday)
            user_id: Optional user filter
            
        Returns:
            Created synthesis entry or None
        """
        if week_start is None:
            today = datetime.utcnow().date()
            week_start = today - timedelta(days=today.weekday() + 1)
        
        week_end = week_start + timedelta(days=6)
        
        # Fetch week's journal entries
        journals = await self._get_weeks_journals(
            agent_id=agent_id,
            tenant_id=tenant_id,
            week_start=week_start,
            week_end=week_end,
            user_id=user_id,
        )
        
        if not journals:
            return None
        
        # Generate synthesis
        synthesis_content = await self._generate_synthesis(journals, agent_id, tenant_id)
        
        # Identify facts to promote
        facts_to_promote = await self._identify_promotable_facts(journals)
        
        # Promote facts to semantic memory
        for fact in facts_to_promote:
            await self._promote_to_memory(
                agent_id=agent_id,
                tenant_id=tenant_id,
                fact=fact,
                user_id=user_id,
            )
        
        # Create synthesis entry
        entry = await self._create_synthesis_entry(
            agent_id=agent_id,
            tenant_id=tenant_id,
            user_id=user_id,
            content=synthesis_content,
            promoted_facts=facts_to_promote,
            source_journals=journals,
            week_start=week_start,
            week_end=week_end,
        )
        
        return entry
    
    async def _get_weeks_journals(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        week_start: date,
        week_end: date,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch daily journal entries for the week."""
        query = """
            SELECT id, content, extracted_facts, emotional_valence, 
                   significance_score, created_at
            FROM journal_entries
            WHERE agent_id = :agent_id
            AND entry_type = 'daily_reflection'
            AND DATE(created_at) >= :week_start
            AND DATE(created_at) <= :week_end
        """
        params: Dict[str, Any] = {
            "agent_id": str(agent_id),
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
        }
        
        if user_id:
            query += " AND (user_id = :user_id OR user_id IS NULL)"
            params["user_id"] = user_id
        
        query += " ORDER BY created_at"
        
        rows = await self.db.fetch_all(query, params, tenant_id)
        
        journals = []
        for row in rows:
            facts = row.get("extracted_facts", "[]")
            if isinstance(facts, str):
                facts = json.loads(facts)
            
            journals.append({
                "id": UUID(row["id"]),
                "content": row["content"],
                "extracted_facts": facts,
                "emotional_valence": row.get("emotional_valence", 0),
                "significance": row.get("significance_score", 0.5),
                "created_at": row["created_at"],
            })
        
        return journals
    
    async def _generate_synthesis(
        self,
        journals: List[Dict],
        agent_id: UUID,
        tenant_id: UUID,
    ) -> str:
        """Generate weekly synthesis content."""
        if self.llm:
            return await self._llm_generate_synthesis(journals, agent_id, tenant_id)
        return self._fallback_generate_synthesis(journals)
    
    async def _llm_generate_synthesis(
        self,
        journals: List[Dict],
        agent_id: UUID,
        tenant_id: UUID,
    ) -> str:
        """Generate synthesis using LLM."""
        journal_texts = "\n\n---\n\n".join([
            f"Day {i+1}:\n{j['content'][:500]}"
            for i, j in enumerate(journals)
        ])
        
        prompt = f"""Review this week's daily journal entries and create a weekly synthesis.

Daily Journals:
{journal_texts}

Synthesize into a weekly reflection (200-300 words) covering:
1. Recurring patterns and themes
2. Relationship developments
3. Facts that appeared multiple times (high confidence)
4. Any contradictions noticed
5. Areas for focus next week

Be specific and actionable."""

        try:
            response = await self.llm.chat.completions.create(
                model="fast",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception:
            return self._fallback_generate_synthesis(journals)
    
    def _fallback_generate_synthesis(self, journals: List[Dict]) -> str:
        """Fallback synthesis without LLM."""
        lines = [f"Weekly Synthesis - {len(journals)} journal entries\n"]
        
        # Aggregate emotional valence
        valences = [j.get("emotional_valence", 0) for j in journals]
        avg_valence = sum(valences) / len(valences) if valences else 0
        
        mood = "positive" if avg_valence > 0.2 else "negative" if avg_valence < -0.2 else "neutral"
        lines.append(f"Overall mood: {mood} ({avg_valence:.2f})")
        
        # Aggregate facts
        all_facts = []
        for j in journals:
            all_facts.extend(j.get("extracted_facts", []))
        
        lines.append(f"\nFacts extracted: {len(all_facts)}")
        
        # Group by type
        fact_types = {}
        for fact in all_facts:
            ftype = fact.get("type", "other")
            if ftype not in fact_types:
                fact_types[ftype] = []
            fact_types[ftype].append(fact)
        
        for ftype, facts in fact_types.items():
            lines.append(f"\n{ftype.title()}: {len(facts)} items")
        
        return "\n".join(lines)
    
    async def _identify_promotable_facts(self, journals: List[Dict]) -> List[Dict]:
        """Identify facts that should be promoted to semantic memory.
        
        Facts are promotable if they:
        - Appear in 2+ journals (recurring)
        - Have high confidence
        - Are not already in memory
        """
        # Collect all facts with content
        all_facts = []
        for j in journals:
            for fact in j.get("extracted_facts", []):
                if "content" in fact:
                    all_facts.append({
                        **fact,
                        "source_journal": j["id"],
                    })
        
        # Simple deduplication and counting
        fact_counts = {}
        for fact in all_facts:
            content = fact.get("content", "")[:100].lower()
            if content not in fact_counts:
                fact_counts[content] = {
                    "content": fact.get("content"),
                    "type": fact.get("type", "fact"),
                    "confidence": fact.get("confidence", 0.5),
                    "count": 0,
                    "sources": [],
                }
            fact_counts[content]["count"] += 1
            fact_counts[content]["sources"].append(fact.get("source_journal"))
            fact_counts[content]["confidence"] = max(
                fact_counts[content]["confidence"],
                fact.get("confidence", 0.5),
            )
        
        # Promote facts that appear 2+ times or have high confidence
        promotable = []
        for key, fact_data in fact_counts.items():
            if fact_data["count"] >= 2 or fact_data["confidence"] >= 0.8:
                promotable.append({
                    "content": fact_data["content"],
                    "type": fact_data["type"],
                    "confidence": min(1.0, fact_data["confidence"] + 0.1 * fact_data["count"]),
                    "evidence_count": fact_data["count"],
                })
        
        return promotable
    
    async def _promote_to_memory(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        fact: Dict,
        user_id: Optional[str] = None,
    ) -> None:
        """Promote a fact to semantic memory.
        
        If similar fact exists, strengthen it instead of duplicating.
        """
        content = fact.get("content", "")
        
        # Check for existing similar memory
        existing = await self.memory_store.get_for_agent(
            agent_id=agent_id,
            tenant_id=tenant_id,
            user_id=user_id,
            limit=100,
        )
        
        # Simple similarity check (production would use embeddings)
        for mem in existing:
            if self._text_similarity(content, mem.content) > 0.8:
                # Strengthen existing memory
                await self.memory_store.update_confidence(
                    memory_id=mem.id,
                    confidence_delta=0.1,
                    tenant_id=tenant_id,
                )
                await self.memory_store.update_access(
                    memory_id=mem.id,
                    tenant_id=tenant_id,
                )
                return
        
        # Create new memory block
        block_type = self._map_fact_type(fact.get("type", "fact"))
        await self.memory_store.create(
            tenant_id=tenant_id,
            agent_id=agent_id,
            user_id=user_id,
            block_type=block_type,
            content=content,
            importance=0.6,  # Promoted facts start with higher importance
            confidence=fact.get("confidence", 0.7),
        )
    
    def _text_similarity(self, a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _map_fact_type(self, fact_type: str) -> MemoryBlockType:
        """Map extracted fact type to memory block type."""
        mapping = {
            "preference": MemoryBlockType.PREFERENCE,
            "identity": MemoryBlockType.FACT,
            "occupation": MemoryBlockType.FACT,
            "belief": MemoryBlockType.BELIEF,
            "relationship": MemoryBlockType.RELATIONSHIP,
        }
        return mapping.get(fact_type, MemoryBlockType.FACT)
    
    async def _create_synthesis_entry(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        content: str,
        promoted_facts: List[Dict],
        source_journals: List[Dict],
        week_start: date,
        week_end: date,
        user_id: Optional[str] = None,
    ) -> JournalEntry:
        """Create and store weekly synthesis entry."""
        from uuid import uuid4
        
        avg_significance = sum(j.get("significance", 0.5) for j in source_journals) / len(source_journals)
        avg_valence = sum(j.get("emotional_valence", 0) for j in source_journals) / len(source_journals)
        
        entry = JournalEntry(
            id=uuid4(),
            tenant_id=tenant_id,
            agent_id=agent_id,
            user_id=user_id,
            entry_type=JournalEntryType.WEEKLY_SYNTHESIS,
            content=content,
            extracted_facts=promoted_facts,
            emotional_valence=avg_valence,
            significance_score=avg_significance,
            source_conversation_ids=[],  # Weekly uses journal sources, not conversations
            source_message_range={
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "journal_count": len(source_journals),
            },
        )
        
        async with self.db.session(tenant_id) as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    INSERT INTO journal_entries
                    (id, tenant_id, agent_id, user_id, entry_type, content,
                     extracted_facts, emotional_valence, significance_score,
                     source_conversation_ids, source_message_range, created_at)
                    VALUES (:id, :tenant_id, :agent_id, :user_id, :entry_type, :content,
                            :extracted_facts, :emotional_valence, :significance_score,
                            :source_ids, :source_range, :created_at)
                """),
                {
                    "id": str(entry.id),
                    "tenant_id": str(entry.tenant_id),
                    "agent_id": str(entry.agent_id),
                    "user_id": entry.user_id,
                    "entry_type": entry.entry_type.value,
                    "content": entry.content,
                    "extracted_facts": json.dumps(entry.extracted_facts),
                    "emotional_valence": entry.emotional_valence,
                    "significance_score": entry.significance_score,
                    "source_ids": json.dumps([]),
                    "source_range": json.dumps(entry.source_message_range),
                    "created_at": entry.created_at.isoformat(),
                },
            )
        
        return entry
