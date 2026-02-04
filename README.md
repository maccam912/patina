# Patina

A biologically-inspired AI agent memory framework that mirrors human cognition—from working memory limits to sleep-like consolidation cycles.

## Features

- **Three-tier memory hierarchy**: Working memory (context window) → Episodic memory (conversations) → Semantic memory (facts)
- **Consolidation cycles**: Daily journaling, weekly synthesis, monthly integration
- **Forgetting curve**: Ebbinghaus-inspired decay (R = e^(-t/S))
- **Multi-tenant**: PostgreSQL with row-level security
- **Provider-agnostic**: LiteLLM for any LLM provider

## Installation

```bash
pip install -e .
```

## Quick Start: Long-Running Agent Mode

The primary way to use Patina is as a persistent, always-listening agent with memory:

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-api-key"

# Start the agent REPL
patina
```

Or with custom options:

```bash
patina --name "My Assistant" --database sqlite+aiosqlite:///mydata.db
```

### REPL Commands

| Command      | Description                        |
|--------------|-------------------------------------|
| `/memories`  | List stored memories               |
| `/clear`     | Start a new conversation           |
| `/status`    | Show agent status                  |
| `/consolidate` | Run manual memory consolidation  |
| `/quit`      | Exit the REPL                      |

### Environment Variables

| Variable             | Description                          |
|----------------------|--------------------------------------|
| `PATINA_DATABASE_URL` | Database connection string          |
| `PATINA_LLM_API_KEY`  | LLM API key (or use provider keys)  |
| `PATINA_LLM_MODEL`    | Model to use                        |
| `ANTHROPIC_API_KEY`   | Anthropic API key                   |
| `OPENAI_API_KEY`      | OpenAI API key                      |

## Programmatic Usage

You can also use Patina as a library:

```python
import asyncio
from patina import PatinaAgent, PatinaConfig

async def main():
    config = PatinaConfig(
        database_url="postgresql+asyncpg://localhost/patina",
        llm_api_key="your-api-key",
    )
    
    agent = await PatinaAgent.create(
        config=config,
        name="Assistant",
        system_prompt="You are a helpful assistant.",
    )
    
    response = await agent.chat("Hello!")
    print(response)

asyncio.run(main())
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
