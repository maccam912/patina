# Patina

A biologically-inspired AI agent memory framework that mirrors human cognition—from working memory limits to sleep-like consolidation cycles.

## Features

- **Three-tier memory hierarchy**: Working memory (context window) → Episodic memory (conversations) → Semantic memory (facts)
- **Consolidation cycles**: Daily journaling, weekly synthesis, monthly integration
- **Forgetting curve**: Ebbinghaus-inspired decay (R = e^(-t/S))
- **Multi-tenant**: PostgreSQL with row-level security
- **Provider-agnostic**: LiteLLM for any LLM provider (OpenRouter, Anthropic, OpenAI, etc.)

## Installation

```bash
pip install -e .
```

## Quick Start: Long-Running Agent Mode

The primary way to use Patina is as a persistent, always-listening agent with memory:

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-openrouter-key"

# Start the agent REPL with an OpenRouter model
patina --model openrouter/anthropic/claude-3.5-sonnet
```

Or with custom options:

```bash
patina \
  --name "My Assistant" \
  --model openrouter/meta-llama/llama-3.1-70b-instruct \
  --database sqlite+aiosqlite:///mydata.db
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

| Variable               | Description                          |
|------------------------|--------------------------------------|
| `PATINA_DATABASE_URL`  | Database connection string          |
| `PATINA_LLM_MODEL`     | Model to use (e.g., `openrouter/anthropic/claude-3.5-sonnet`) |
| `OPENROUTER_API_KEY`   | OpenRouter API key                   |
| `ANTHROPIC_API_KEY`    | Anthropic API key (direct)           |
| `OPENAI_API_KEY`       | OpenAI API key (direct)              |

---

## Running as a System Service (systemd)

Create a systemd unit file at `/etc/systemd/system/patina.service`:

```ini
[Unit]
Description=Patina AI Agent
After=network.target postgresql.service

[Service]
Type=simple
User=patina
WorkingDirectory=/opt/patina
Environment="OPENROUTER_API_KEY=your-openrouter-key"
Environment="PATINA_DATABASE_URL=postgresql+asyncpg://localhost/patina"
Environment="PATINA_LLM_MODEL=openrouter/anthropic/claude-3.5-sonnet"
ExecStart=/opt/patina/.venv/bin/patina --no-scheduler
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable patina
sudo systemctl start patina
```

---

## Telegram Bot Integration

Build a Telegram bot that uses Patina for persistent memory:

```python
import asyncio
import os
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from patina import PatinaAgent, PatinaConfig

# Global agent instance
agent = None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    message = update.message.text
    
    # Chat with memory, keyed by Telegram user ID
    response = await agent.chat(message, user_id=user_id)
    await update.message.reply_text(response)

async def main():
    global agent
    
    # Initialize Patina agent with OpenRouter
    config = PatinaConfig(
        llm_model="openrouter/anthropic/claude-3.5-sonnet",
    )
    agent = await PatinaAgent.create(
        config=config,
        name="TelegramBot",
        system_prompt="You are a helpful assistant with persistent memory.",
    )
    
    # Build Telegram bot
    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Run both the bot and background consolidation
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
```

Install dependencies: `pip install python-telegram-bot`

Set environment variables:
```bash
export OPENROUTER_API_KEY="your-openrouter-key"
export TELEGRAM_BOT_TOKEN="your-telegram-token"
python telegram_bot.py
```

---

## Custom Tools with Agentic Loop

Patina supports an agentic loop where the model calls tools until it produces a final text response:

```python
import asyncio
import json
import os
from patina import PatinaAgent, PatinaConfig

# Define custom tools
CUSTOM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]

async def execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return result."""
    if name == "get_weather":
        # Mock weather API
        return json.dumps({"temp": 72, "condition": "sunny", "location": args["location"]})
    elif name == "send_email":
        # Mock email send
        return f"Email sent to {args['to']}"
    return "Unknown tool"

async def agentic_loop(agent: PatinaAgent, user_message: str, max_iterations: int = 10):
    """Run agent loop until text response or max iterations."""
    
    # Get memory tools + custom tools
    from patina.llm.tools import get_memory_tools
    all_tools = get_memory_tools() + CUSTOM_TOOLS
    
    # Build initial context
    context = await agent.memory.build_context(
        agent={
            "id": agent.agent_id,
            "system_prompt": agent._agent_data.get("system_prompt", ""),
            "persona_block": agent.persona,
            "human_block": agent.human_block,
        },
        conversation_id=agent._current_conversation.id if agent._current_conversation else None,
        tenant_id=agent.tenant_id,
        current_query=user_message,
    )
    context.append({"role": "user", "content": user_message})
    
    for _ in range(max_iterations):
        response = await agent.llm.completion(messages=context, tools=all_tools)
        message = response.choices[0].message
        
        # No tool calls = final response
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return message.content
        
        # Add assistant message with tool calls
        context.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]
        })
        
        # Execute each tool
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = await execute_tool(tc.function.name, args)
            context.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    
    return "Max iterations reached"

async def main():
    # Use OpenRouter with a capable model
    config = PatinaConfig(
        llm_model="openrouter/anthropic/claude-3.5-sonnet",
    )
    agent = await PatinaAgent.create(
        config=config,
        system_prompt="You can check weather and send emails. Use tools when needed.",
    )
    await agent.start_conversation()
    
    # Agent will call tools as needed
    response = await agentic_loop(agent, "What's the weather in Tokyo? Then email me a summary.")
    print(response)

if __name__ == "__main__":
    # Requires: export OPENROUTER_API_KEY="your-key"
    asyncio.run(main())
```

The agentic loop:
1. Sends the user message with all available tools
2. If the model returns tool calls, executes them and adds results to context
3. Loops until the model returns a text-only response
4. Memory tools (`memory_search`, `memory_store`) are automatically included

---

## Programmatic Usage

Basic library usage for simple integrations:

```python
import asyncio
from patina import PatinaAgent, PatinaConfig

async def main():
    # Configure with OpenRouter
    config = PatinaConfig(
        database_url="postgresql+asyncpg://localhost/patina",
        llm_model="openrouter/anthropic/claude-3.5-sonnet",
    )
    
    agent = await PatinaAgent.create(
        config=config,
        name="Assistant",
        system_prompt="You are a helpful assistant.",
    )
    
    response = await agent.chat("Hello!")
    print(response)

if __name__ == "__main__":
    # Requires: export OPENROUTER_API_KEY="your-key"
    asyncio.run(main())
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
