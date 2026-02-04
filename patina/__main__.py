"""CLI entry point for Patina long-running agent mode.

Usage:
    python -m patina
    patina  # if installed with pip install -e .

Environment variables:
    PATINA_DATABASE_URL: Database connection string
    PATINA_LLM_API_KEY: LLM API key (or provider-specific keys)
    PATINA_LLM_MODEL: Model to use (default: anthropic/claude-sonnet-4-20250514)
"""

import asyncio
import argparse
import signal
import sys
import warnings
from typing import Optional

# Suppress harmless Pydantic serialization warnings from LiteLLM
# These occur when LLM providers return slightly different response formats
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
    module="pydantic.*",
)

from patina.config import PatinaConfig
from patina.agent.core import PatinaAgent
from patina.scheduling.jobs import ConsolidationScheduler
from patina.consolidation.consolidator import MemoryConsolidator


class PatinaREPL:
    """Interactive REPL for Patina agent."""
    
    def __init__(self, agent: PatinaAgent, scheduler: Optional[ConsolidationScheduler] = None):
        self.agent = agent
        self.scheduler = scheduler
        self.running = False
    
    async def start(self) -> None:
        """Start the REPL loop."""
        self.running = True
        print(f"\n🧠 Patina Agent [{self.agent.name}] ready")
        print("Type your message, or use commands: /quit, /memories, /clear, /help")
        print("-" * 50)
        
        # Start background scheduler if available
        if self.scheduler:
            await self.scheduler.start([self.agent.agent_id])
            print("📅 Background consolidation scheduler started")
        
        while self.running:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\nYou: ").strip()
                )
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue
                
                # Chat with agent
                print("\nAssistant: ", end="", flush=True)
                response = await self.agent.chat(user_input)
                print(response)
                
            except EOFError:
                # Handle Ctrl+D
                await self.stop()
            except KeyboardInterrupt:
                # Handle Ctrl+C
                await self.stop()
    
    async def _handle_command(self, command: str) -> None:
        """Handle REPL commands."""
        parts = command.lower().split()
        cmd = parts[0]
        
        if cmd in ("/quit", "/exit", "/q"):
            await self.stop()
        
        elif cmd == "/help":
            print("""
Commands:
  /quit, /exit, /q  - Exit the REPL
  /memories         - List stored memories
  /clear            - Start a new conversation
  /status           - Show agent status
  /consolidate      - Run manual memory consolidation
  /help             - Show this help message
""")
        
        elif cmd == "/memories":
            memories = await self.agent.get_memories(limit=10)
            if not memories:
                print("No memories stored yet.")
            else:
                print(f"\n📚 Recent memories ({len(memories)}):")
                for m in memories:
                    print(f"  [{m['type']}] {m['content'][:80]}...")
        
        elif cmd == "/clear":
            await self.agent.start_conversation()
            print("🆕 Started new conversation.")
        
        elif cmd == "/status":
            print(f"""
Agent Status:
  Name: {self.agent.name}
  ID: {self.agent.agent_id}
  Persona: {self.agent.persona[:50] + '...' if self.agent.persona else '(none)'}
  Scheduler: {'running' if self.scheduler else 'disabled'}
""")
        
        elif cmd == "/consolidate":
            if self.scheduler:
                print("Running manual consolidation...")
                result = await self.scheduler.run_manual(
                    agent_id=self.agent.agent_id,
                    tenant_id=self.agent.tenant_id,
                    job_type="daily",
                )
                print(f"Consolidation complete: {result}")
            else:
                print("Scheduler not available.")
        
        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")
    
    async def stop(self) -> None:
        """Stop the REPL."""
        self.running = False
        print("\n👋 Goodbye!")
        
        if self.scheduler:
            await self.scheduler.stop()
        
        await self.agent.close()


async def run_agent(args: argparse.Namespace) -> None:
    """Run the agent in REPL mode."""
    # Build config from args and environment
    config = PatinaConfig(
        database_url=args.database or None,
        llm_model=args.model or None,
    )
    
    # Create agent
    agent = await PatinaAgent.create(
        config=config,
        name=args.name,
        system_prompt=args.system_prompt,
    )
    
    # Create scheduler if consolidation enabled
    scheduler = None
    if config.consolidation_enabled and not args.no_scheduler:
        consolidator = MemoryConsolidator(agent.db, agent.llm.chat)
        scheduler = ConsolidationScheduler(agent.db, consolidator)
    
    # Run REPL
    repl = PatinaREPL(agent, scheduler)
    await repl.start()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Patina - Biologically-inspired AI agent memory framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  patina                           # Start with defaults
  patina --name "My Assistant"     # Custom agent name
  patina --database sqlite+aiosqlite:///mydata.db

Environment variables:
  PATINA_DATABASE_URL              Database connection string
  PATINA_LLM_API_KEY               LLM API key
  ANTHROPIC_API_KEY                Anthropic API key
  OPENAI_API_KEY                   OpenAI API key
""",
    )
    
    parser.add_argument(
        "--name", "-n",
        default="Patina",
        help="Agent name (default: Patina)",
    )
    parser.add_argument(
        "--database", "-d",
        default=None,
        help="Database URL (default: sqlite+aiosqlite:///patina.db)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model to use (default: anthropic/claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--system-prompt", "-s",
        default="You are a helpful AI assistant with persistent memory.",
        help="System prompt for the agent",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable background consolidation scheduler",
    )
    
    args = parser.parse_args()
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the agent
    try:
        asyncio.run(run_agent(args))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
