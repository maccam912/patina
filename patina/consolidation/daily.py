"""Daily consolidation: Journal generation from conversations.

Implements the daily 'sleep replay' that processes the day's conversations
and creates reflective journal entries.
"""

from datetime import datetime, date
from typing import List, Dict, Optional, Any
from uuid import UUID
import json

from patina.database.connection import Database
from patina.database.models import JournalEntry, JournalEntryType
from patina.memory.episodic import ConversationStore, MessageStore


class DailyConsolidator:
    """Daily journal generation from conversations.
    
    Mirrors the hippocampal replay during sleep that consolidates
    episodic memories into more stable representations.
    """
    
    def __init__(self, db: Database, llm_client=None):
        self.db = db
        self.llm = llm_client
        self.conversations = ConversationStore(db)
        self.messages = MessageStore(db)
    
    async def run(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        target_date: Optional[date] = None,
        user_id: Optional[str] = None,
    ) -> Optional[JournalEntry]:
        """Generate daily journal entry from conversations.
        
        Args:
            agent_id: Agent to consolidate for
            tenant_id: Tenant context
            target_date: Date to consolidate (defaults to yesterday)
            user_id: Optional user filter
            
        Returns:
            Created journal entry or None if no conversations
        """
        target_date = target_date or (datetime.utcnow().date())
        
        # Fetch day's conversations
        conversations = await self._get_days_conversations(
            agent_id=agent_id,
            tenant_id=tenant_id,
            target_date=target_date,
            user_id=user_id,
        )
        
        if not conversations:
            return None
        
        # Generate journal content
        journal_content = await self._generate_journal(conversations, agent_id, tenant_id)
        
        # Extract structured facts
        extracted_facts = await self._extract_facts(conversations)
        
        # Calculate emotional valence
        emotional_valence = self._calculate_emotional_valence(conversations)
        
        # Calculate significance
        significance = self._calculate_significance(conversations)
        
        # Store journal entry
        entry = await self._create_journal_entry(
            agent_id=agent_id,
            tenant_id=tenant_id,
            user_id=user_id,
            content=journal_content,
            extracted_facts=extracted_facts,
            emotional_valence=emotional_valence,
            significance=significance,
            source_ids=[c["id"] for c in conversations],
            target_date=target_date,
        )
        
        # Mark source messages as summarized
        await self._mark_summarized(conversations, tenant_id)
        
        return entry
    
    async def _get_days_conversations(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        target_date: date,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch all conversations and messages for a given date."""
        # Get conversations created on target date
        query = """
            SELECT c.id, c.user_id, c.title, c.created_at
            FROM conversations c
            WHERE c.agent_id = :agent_id
            AND DATE(c.created_at) = :target_date
        """
        params: Dict[str, Any] = {
            "agent_id": str(agent_id),
            "target_date": target_date.isoformat(),
        }
        
        if user_id:
            query += " AND c.user_id = :user_id"
            params["user_id"] = user_id
        
        conv_rows = await self.db.fetch_all(query, params, tenant_id)
        
        # Also get conversations with messages on target date
        query2 = """
            SELECT DISTINCT c.id, c.user_id, c.title, c.created_at
            FROM conversations c
            JOIN messages m ON m.conversation_id = c.id
            WHERE c.agent_id = :agent_id
            AND DATE(m.created_at) = :target_date
        """
        more_rows = await self.db.fetch_all(query2, params, tenant_id)
        
        # Combine and dedupe
        seen_ids = set()
        conversations = []
        
        for row in conv_rows + more_rows:
            conv_id = row["id"]
            if conv_id not in seen_ids:
                seen_ids.add(conv_id)
                
                # Get messages for this conversation on target date
                msg_rows = await self.db.fetch_all(
                    """
                    SELECT id, role, content, importance_score, created_at
                    FROM messages
                    WHERE conversation_id = :conv_id
                    AND DATE(created_at) = :target_date
                    ORDER BY created_at
                    """,
                    {"conv_id": conv_id, "target_date": target_date.isoformat()},
                    tenant_id,
                )
                
                if msg_rows:
                    conversations.append({
                        "id": UUID(conv_id),
                        "user_id": row.get("user_id"),
                        "title": row.get("title"),
                        "messages": [
                            {
                                "id": UUID(m["id"]),
                                "role": m["role"],
                                "content": m["content"],
                                "importance": m.get("importance_score", 0.5),
                            }
                            for m in msg_rows
                        ],
                    })
        
        return conversations
    
    async def _generate_journal(
        self,
        conversations: List[Dict],
        agent_id: UUID,
        tenant_id: UUID,
    ) -> str:
        """Generate journal content using LLM or fallback."""
        if self.llm:
            return await self._llm_generate_journal(conversations, agent_id, tenant_id)
        return self._fallback_generate_journal(conversations)
    
    async def _llm_generate_journal(
        self,
        conversations: List[Dict],
        agent_id: UUID,
        tenant_id: UUID,
    ) -> str:
        """Generate journal using LLM."""
        # Get agent persona for context
        agent_row = await self.db.fetch_one(
            "SELECT persona_block, human_block FROM agents WHERE id = :id",
            {"id": str(agent_id)},
            tenant_id,
        )
        
        persona = agent_row.get("persona_block", "") if agent_row else ""
        human = agent_row.get("human_block", "") if agent_row else ""
        
        prompt = f"""You are maintaining a personal journal for an AI assistant.
Review today's conversations and write a reflective journal entry.

Your current persona:
{persona[:500] if persona else "Not specified"}

Your understanding of the user(s):
{human[:500] if human else "Not specified"}

Today's conversations:
{self._format_conversations(conversations)}

Write a journal entry reflecting on today (200-400 words). Include:
1. Key learnings about the user(s)
2. Significant moments or decisions
3. Patterns noticed
4. Open loops/follow-ups needed
5. Self-observations

Write in first person. Be specific with names and details."""

        try:
            response = await self.llm.chat.completions.create(
                model="fast",  # Use fast model for consolidation
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._fallback_generate_journal(conversations)
    
    def _fallback_generate_journal(self, conversations: List[Dict]) -> str:
        """Fallback journal generation without LLM."""
        lines = [f"Daily reflection - {len(conversations)} conversation(s)\n"]
        
        for conv in conversations:
            user = conv.get("user_id") or "Unknown user"
            msg_count = len(conv.get("messages", []))
            lines.append(f"\n## Conversation with {user}")
            lines.append(f"Messages: {msg_count}")
            
            # Extract key messages (high importance)
            important = [
                m for m in conv.get("messages", [])
                if m.get("importance", 0) > 0.6
            ]
            if important:
                lines.append("Key exchanges:")
                for msg in important[:3]:
                    content = msg.get("content", "")[:100]
                    lines.append(f"- [{msg.get('role')}]: {content}...")
        
        return "\n".join(lines)
    
    def _format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversations for LLM prompt."""
        parts = []
        for conv in conversations:
            user = conv.get("user_id") or "user"
            parts.append(f"### Conversation with {user}")
            
            for msg in conv.get("messages", [])[:20]:  # Limit messages
                role = msg.get("role", "user")
                content = msg.get("content", "")[:300]
                parts.append(f"{role}: {content}")
            
            parts.append("")
        
        return "\n".join(parts)
    
    async def _extract_facts(self, conversations: List[Dict]) -> List[Dict]:
        """Extract structured facts from conversations."""
        facts = []
        
        for conv in conversations:
            for msg in conv.get("messages", []):
                content = msg.get("content", "").lower()
                
                # Simple heuristic fact extraction
                # In production, this would use LLM
                if "my name is" in content or "i'm " in content or "i am " in content:
                    facts.append({
                        "type": "identity",
                        "content": msg.get("content", "")[:200],
                        "confidence": 0.8,
                    })
                elif "i prefer" in content or "i like" in content or "i love" in content:
                    facts.append({
                        "type": "preference",
                        "content": msg.get("content", "")[:200],
                        "confidence": 0.7,
                    })
                elif "i work" in content or "my job" in content:
                    facts.append({
                        "type": "occupation",
                        "content": msg.get("content", "")[:200],
                        "confidence": 0.8,
                    })
        
        return facts
    
    def _calculate_emotional_valence(self, conversations: List[Dict]) -> float:
        """Calculate overall emotional valence (-1 to 1)."""
        positive_words = ["great", "thank", "love", "excellent", "perfect", "happy", "glad"]
        negative_words = ["bad", "wrong", "hate", "terrible", "frustrated", "angry", "sad"]
        
        positive_count = 0
        negative_count = 0
        
        for conv in conversations:
            for msg in conv.get("messages", []):
                content = msg.get("content", "").lower()
                for word in positive_words:
                    if word in content:
                        positive_count += 1
                for word in negative_words:
                    if word in content:
                        negative_count += 1
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def _calculate_significance(self, conversations: List[Dict]) -> float:
        """Calculate overall significance score."""
        if not conversations:
            return 0.0
        
        total_messages = sum(len(c.get("messages", [])) for c in conversations)
        avg_importance = 0.5
        
        all_importances = []
        for conv in conversations:
            for msg in conv.get("messages", []):
                all_importances.append(msg.get("importance", 0.5))
        
        if all_importances:
            avg_importance = sum(all_importances) / len(all_importances)
        
        # Scale by message count and importance
        return min(1.0, (total_messages / 20) * 0.5 + avg_importance * 0.5)
    
    async def _create_journal_entry(
        self,
        agent_id: UUID,
        tenant_id: UUID,
        content: str,
        extracted_facts: List[Dict],
        emotional_valence: float,
        significance: float,
        source_ids: List[UUID],
        target_date: date,
        user_id: Optional[str] = None,
    ) -> JournalEntry:
        """Create and store journal entry."""
        from uuid import uuid4
        
        entry = JournalEntry(
            id=uuid4(),
            tenant_id=tenant_id,
            agent_id=agent_id,
            user_id=user_id,
            entry_type=JournalEntryType.DAILY_REFLECTION,
            content=content,
            extracted_facts=extracted_facts,
            emotional_valence=emotional_valence,
            significance_score=significance,
            source_conversation_ids=source_ids,
            source_message_range={
                "date": target_date.isoformat(),
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
                    "source_ids": json.dumps([str(x) for x in entry.source_conversation_ids]),
                    "source_range": json.dumps(entry.source_message_range),
                    "created_at": entry.created_at.isoformat(),
                },
            )
        
        return entry
    
    async def _mark_summarized(
        self,
        conversations: List[Dict],
        tenant_id: UUID,
    ) -> None:
        """Mark messages as having been summarized."""
        message_ids = []
        for conv in conversations:
            for msg in conv.get("messages", []):
                if "id" in msg:
                    message_ids.append(msg["id"])
        
        if message_ids:
            await self.messages.mark_summarized(message_ids, tenant_id)
