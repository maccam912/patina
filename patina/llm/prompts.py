"""Consolidation prompt templates."""

DAILY_JOURNAL_PROMPT = """You are {agent_name}, reflecting on your conversations from {date}.

Your current understanding of yourself (persona):
{persona_block}

Your current understanding of the user(s):
{human_block}

Today's conversations:
{formatted_conversations}

Write a journal entry reflecting on today. Include:

1. **Key Learnings**: What new information did you learn about the user(s)?
   - Preferences, habits, opinions expressed
   - Personal details shared (family, work, interests)
   - Communication style observations

2. **Significant Moments**: What interactions felt important?
   - Decisions made together
   - Emotional exchanges (frustration, joy, gratitude)
   - Commitments or promises made

3. **Patterns Noticed**: Any recurring themes?
   - Topics that came up multiple times
   - Questions asked repeatedly
   - Behavioral patterns

4. **Open Loops**: What remains unresolved?
   - Questions you couldn't answer
   - Tasks promised but not completed
   - Topics to follow up on

5. **Self-Observations**: How did you perform?
   - What went well?
   - What could improve?
   - Any inconsistencies in your responses?

Write in first person. Be specific—use names, dates, details.
Target length: 300-500 words.

After your reflection, provide a structured extraction:
```json
{{
  "facts_learned": [
    {{"subject": "user_name", "predicate": "prefers", "object": "...", "confidence": 0.9}}
  ],
  "commitments": ["..."],
  "follow_ups": ["..."],
  "emotional_summary": "positive/neutral/negative",
  "importance_score": 0.0-1.0
}}
```"""


WEEKLY_SYNTHESIS_PROMPT = """You are {agent_name}, synthesizing your week of {week_start} to {week_end}.

This week's daily journal entries:
{journal_entries}

Current memory blocks about users:
{current_user_memories}

Perform a weekly synthesis:

1. **Recurring Patterns**: What themes appeared multiple times this week?
   - User interests that persisted
   - Types of requests that repeated
   - Behavioral consistencies

2. **Relationship Evolution**: How has your understanding of users deepened?
   - New dimensions of their personality revealed
   - Trust indicators (sharing more personal info, etc.)
   - Communication style adaptations working/not working

3. **Fact Consolidation**: Which observations are now confident enough to remember long-term?
   - Facts mentioned 2+ times
   - Facts that explain multiple behaviors
   - Facts with strong emotional association

4. **Contradictions**: Any inconsistencies to resolve?
   - User said X on Monday but Y on Thursday
   - Your behavior varied unexpectedly
   - Preferences that seem contradictory

5. **Growth Areas**: What should you focus on next week?
   - Skills to develop
   - Topics to learn more about
   - Relationship aspects to nurture

Output format:
```json
{{
  "patterns": ["..."],
  "facts_to_promote": [
    {{
      "content": "User prefers concise responses",
      "confidence": 0.85,
      "evidence_count": 3,
      "block_type": "preference"
    }}
  ],
  "contradictions": [
    {{"fact_a": "...", "fact_b": "...", "resolution": "..."}}
  ],
  "deprecated_facts": ["memory_block_ids to mark as superseded"],
  "focus_areas": ["..."]
}}
```"""


MONTHLY_INTEGRATION_PROMPT = """You are {agent_name}, performing monthly personality integration for {month}.

Current persona block:
{persona_block}

Current human/user block:
{human_block}

This month's weekly syntheses:
{weekly_syntheses}

All current memory blocks:
{memory_blocks}

Perform deep integration:

1. **Persona Evolution**: Based on this month's experiences, how should your core persona evolve?
   - New capabilities demonstrated
   - Refined understanding of your role
   - Values clarified through interactions
   
   Rewrite your persona block to reflect growth while maintaining core identity.

2. **User Model Update**: Rewrite the human block with consolidated understanding.
   - Merge redundant facts
   - Remove outdated information
   - Highlight most important patterns

3. **Memory Pruning**: Which memory blocks can be:
   - Merged (redundant information)
   - Promoted to persona/human blocks (core enough)
   - Archived (no longer relevant)

4. **Identity Coherence**: Ensure consistency across all memory layers.
   - Does your persona match your behavior patterns?
   - Are user models consistent with observations?
   - Any cognitive dissonance to resolve?

Output:
```json
{{
  "updated_persona_block": "New persona text (max 2000 chars)",
  "updated_human_block": "New human text (max 2000 chars)",
  "memory_operations": [
    {{"action": "merge", "source_ids": ["...", "..."], "merged_content": "..."}},
    {{"action": "archive", "ids": ["..."]}},
    {{"action": "promote_to_persona", "id": "...", "integration_note": "..."}}
  ],
  "identity_notes": "Reflection on growth and consistency"
}}
```"""


def format_daily_prompt(
    agent_name: str,
    date: str,
    persona_block: str,
    human_block: str,
    conversations: str,
) -> str:
    """Format the daily journal prompt."""
    return DAILY_JOURNAL_PROMPT.format(
        agent_name=agent_name,
        date=date,
        persona_block=persona_block or "Not defined",
        human_block=human_block or "Not defined",
        formatted_conversations=conversations,
    )


def format_weekly_prompt(
    agent_name: str,
    week_start: str,
    week_end: str,
    journal_entries: str,
    current_user_memories: str,
) -> str:
    """Format the weekly synthesis prompt."""
    return WEEKLY_SYNTHESIS_PROMPT.format(
        agent_name=agent_name,
        week_start=week_start,
        week_end=week_end,
        journal_entries=journal_entries,
        current_user_memories=current_user_memories,
    )


def format_monthly_prompt(
    agent_name: str,
    month: str,
    persona_block: str,
    human_block: str,
    weekly_syntheses: str,
    memory_blocks: str,
) -> str:
    """Format the monthly integration prompt."""
    return MONTHLY_INTEGRATION_PROMPT.format(
        agent_name=agent_name,
        month=month,
        persona_block=persona_block or "Not defined",
        human_block=human_block or "Not defined",
        weekly_syntheses=weekly_syntheses,
        memory_blocks=memory_blocks,
    )
