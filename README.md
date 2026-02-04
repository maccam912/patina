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

## Quick Start

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
