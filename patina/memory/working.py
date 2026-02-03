"""Working memory / context window management.

Implements working memory limits based on cognitive science research
(Miller's Law / Cowan's 4-7 semantic chunks).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import tiktoken


@dataclass
class ContextBudget:
    """Token budget allocation following working memory principles.
    
    Based on cognitive research showing humans can hold 4-7 semantic chunks
    in working memory at once.
    """
    total_tokens: int = 128000
    output_reserved: int = 4000
    system_prompt: int = 2000
    persona_block: int = 800      # ~1 semantic chunk
    human_block: int = 800        # ~1 semantic chunk
    retrieved_memories: int = 2000  # ~2 semantic chunks
    recent_messages: int = 8000   # ~2-3 semantic chunks
    
    @property
    def available_for_history(self) -> int:
        """Remaining tokens available for conversation history."""
        fixed = (
            self.output_reserved + 
            self.system_prompt + 
            self.persona_block + 
            self.human_block + 
            self.retrieved_memories + 
            self.recent_messages
        )
        return self.total_tokens - fixed


class ContextWindowManager:
    """Manages context window assembly respecting cognitive limits.
    
    Assembles the LLM context from:
    1. System prompt with persona (chunk 1)
    2. Retrieved memories (chunks 2-3)
    3. Conversation history with primacy/recency (chunks 4-7)
    """
    
    def __init__(
        self, 
        budget: Optional[ContextBudget] = None,
        llm_client: Any = None,
    ):
        self.budget = budget or ContextBudget()
        self.llm_client = llm_client
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback for testing without tiktoken data
            self._encoder = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._encoder:
            return len(self._encoder.encode(text))
        # Rough fallback: ~4 chars per token
        return len(text) // 4
    
    def _total_message_tokens(self, messages: List[Dict]) -> int:
        """Count total tokens in message list."""
        return sum(
            self.count_tokens(m.get("content", "")) 
            for m in messages
        )
    
    def _truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        if self._encoder:
            tokens = self._encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text
            # Leave room for ellipsis
            truncated = self._encoder.decode(tokens[:max_tokens - 10])
            return truncated + "\n..."
        # Fallback
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 10] + "\n..."
    
    async def build_context(
        self,
        agent: Dict,
        messages: List[Dict],
        retrieved_memories: List[Dict],
        current_query: Optional[str] = None,
    ) -> List[Dict]:
        """Assemble context respecting working memory limits (4-7 chunks).
        
        Args:
            agent: Agent configuration with system_prompt, persona_block, human_block
            messages: Conversation history
            retrieved_memories: Relevant memories from semantic store
            current_query: Current user query for relevance weighting
            
        Returns:
            List of messages ready for LLM API
        """
        context = []
        
        # Chunk 1: System prompt with persona
        persona = agent.get("persona_block", "")[:self.budget.persona_block * 4]
        human = agent.get("human_block", "")[:self.budget.human_block * 4]
        
        system_parts = [agent.get("system_prompt", "You are a helpful assistant.")]
        
        if persona:
            system_parts.append(f"\nAbout yourself:\n{persona}")
        if human:
            system_parts.append(f"\nAbout the user:\n{human}")
        
        system_content = "\n".join(system_parts)
        context.append({"role": "system", "content": system_content})
        
        # Chunks 2-3: Retrieved memories (semantic, relevant to current query)
        if retrieved_memories:
            memory_lines = [f"- {m.get('content', '')}" for m in retrieved_memories[:10]]
            memory_content = "Relevant memories:\n" + "\n".join(memory_lines)
            
            if self.count_tokens(memory_content) > self.budget.retrieved_memories:
                memory_content = self._truncate(
                    memory_content, 
                    self.budget.retrieved_memories
                )
            
            context.append({
                "role": "system",
                "content": memory_content,
                "name": "memory_context",
            })
        
        # Chunks 4-7: Recent conversation (preserving primacy/recency effects)
        available = self.budget.available_for_history
        
        if self._total_message_tokens(messages) <= available:
            context.extend(messages)
        else:
            # Compress history preserving primacy/recency
            compressed = await self._compress_history(messages, available)
            context.extend(compressed)
        
        return context
    
    async def _compress_history(
        self,
        messages: List[Dict],
        max_tokens: int,
    ) -> List[Dict]:
        """Compress conversation history preserving serial position effects.
        
        Cognitive research shows:
        - Primacy effect: First items remembered well
        - Recency effect: Last items remembered well
        - Middle items forgotten first
        """
        if not messages:
            return []
        
        # Always keep first user message (primacy)
        first_msg = messages[0] if messages else None
        
        # Keep recent messages (recency)
        recent_count = 6
        recent = messages[-recent_count:] if len(messages) > recent_count else messages
        
        # Summarize middle messages
        middle = messages[1:-recent_count] if len(messages) > recent_count + 1 else []
        
        if middle:
            summary = await self._generate_summary(middle)
            summary_msg = {
                "role": "system",
                "content": f"[Earlier conversation summary]\n{summary}",
                "name": "history_summary",
            }
            result = [first_msg, summary_msg] + recent if first_msg else [summary_msg] + recent
        else:
            result = messages
        
        # Final truncation if still over budget
        while self._total_message_tokens(result) > max_tokens and len(result) > 2:
            result.pop(1)  # Remove oldest non-system message
        
        return result
    
    async def _generate_summary(self, messages: List[Dict]) -> str:
        """Generate summary of messages using LLM.
        
        If no LLM client available, creates a simple extractive summary.
        """
        if not messages:
            return ""
        
        if self.llm_client:
            # Use fast model for summarization
            try:
                prompt = "Summarize this conversation excerpt concisely (2-3 sentences):\n\n"
                prompt += "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')[:200]}" 
                    for m in messages
                )
                
                response = await self.llm_client.chat.completions.create(
                    model="fast",  # Uses routed fast model
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                )
                return response.choices[0].message.content
            except Exception:
                pass
        
        # Fallback: simple extractive summary
        summary_parts = []
        for msg in messages[:5]:  # First 5 messages
            content = msg.get("content", "")[:100]
            role = msg.get("role", "user")
            summary_parts.append(f"[{role}]: {content}...")
        
        return "\n".join(summary_parts)
    
    def estimate_context_usage(self, context: List[Dict]) -> Dict[str, int]:
        """Estimate token usage across context sections."""
        usage = {
            "system": 0,
            "memory": 0,
            "history": 0,
            "total": 0,
        }
        
        for msg in context:
            tokens = self.count_tokens(msg.get("content", ""))
            name = msg.get("name", "")
            role = msg.get("role", "")
            
            if role == "system" and name == "memory_context":
                usage["memory"] += tokens
            elif role == "system":
                usage["system"] += tokens
            else:
                usage["history"] += tokens
            
            usage["total"] += tokens
        
        return usage
