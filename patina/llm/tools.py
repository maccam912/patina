"""Tool definitions for agent memory operations."""

from typing import List, Dict


MEMORY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": "Search long-term memory for relevant information about the user or past conversations. Use this when you need to recall something you've learned about the user, their preferences, or previous discussions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory (be specific)",
                },
                "memory_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["fact", "preference", "summary", "belief", "relationship"],
                    },
                    "description": "Types of memories to search (optional, searches all if not specified)",
                },
                "user_id": {
                    "type": "string",
                    "description": "Filter to specific user (optional)",
                },
            },
            "required": ["query"],
        },
    },
}


MEMORY_STORE_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_store",
        "description": "Store important information for long-term memory. Use this when the user shares something worth remembering for future conversations.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "What to remember (be specific and factual)",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["fact", "preference", "belief", "relationship"],
                    "description": "Type of memory to store",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "How important is this (0-1, default 0.5)",
                },
                "user_id": {
                    "type": "string",
                    "description": "User this memory relates to (optional)",
                },
            },
            "required": ["content", "memory_type"],
        },
    },
}


MEMORY_UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_update",
        "description": "Update or correct an existing memory. Use when you learn information that supersedes or corrects something you previously knew.",
        "parameters": {
            "type": "object",
            "properties": {
                "old_content": {
                    "type": "string",
                    "description": "Description of the memory to update",
                },
                "new_content": {
                    "type": "string",
                    "description": "The corrected/updated information",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this memory is being updated",
                },
            },
            "required": ["old_content", "new_content"],
        },
    },
}


# Standard tool set for agents
MEMORY_TOOLS: List[Dict] = [
    MEMORY_SEARCH_TOOL,
    MEMORY_STORE_TOOL,
]

# Extended tool set including update
MEMORY_TOOLS_EXTENDED: List[Dict] = [
    MEMORY_SEARCH_TOOL,
    MEMORY_STORE_TOOL,
    MEMORY_UPDATE_TOOL,
]


def get_memory_tools(include_update: bool = False) -> List[Dict]:
    """Get memory tool definitions.
    
    Args:
        include_update: Whether to include the memory_update tool
        
    Returns:
        List of tool definitions
    """
    if include_update:
        return MEMORY_TOOLS_EXTENDED.copy()
    return MEMORY_TOOLS.copy()


def format_tool_result(
    tool_call_id: str,
    result: str,
    is_error: bool = False,
) -> Dict:
    """Format a tool result for including in messages."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result if not is_error else f"Error: {result}",
    }
