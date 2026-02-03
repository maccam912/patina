"""LLM Router: LiteLLM integration with multi-provider fallback."""

import os
from typing import Optional, List, Dict, Any

try:
    from litellm import Router
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    Router = None

from patina.config import PatinaConfig


class LLMRouter:
    """LiteLLM router with multi-provider fallback and rate limiting.
    
    Provides:
    - Primary model (default: Claude)
    - Fast model for summarization
    - Fallback chain for reliability
    """
    
    def __init__(self, config: Optional[PatinaConfig] = None):
        self.config = config or PatinaConfig()
        self._router = None
        self._mock_mode = False
    
    def _build_model_list(self) -> List[Dict]:
        """Build model configuration list."""
        model_list = []
        
        api_key = self.config.llm_api_key or os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        google_key = os.environ.get("GOOGLE_API_KEY")
        
        # Primary: Default model
        if api_key:
            if "anthropic" in self.config.llm_model:
                model_list.append({
                    "model_name": "default",
                    "litellm_params": {
                        "model": self.config.llm_model,
                        "api_key": api_key,
                        "rpm": 400,
                        "tpm": 100000,
                    }
                })
            elif "openai" in self.config.llm_model:
                model_list.append({
                    "model_name": "default",
                    "litellm_params": {
                        "model": self.config.llm_model,
                        "api_key": api_key,
                        "rpm": 500,
                        "tpm": 150000,
                    }
                })
        
        # Fast model for summarization
        if api_key and "anthropic" in self.config.llm_fast_model:
            model_list.append({
                "model_name": "fast",
                "litellm_params": {
                    "model": self.config.llm_fast_model,
                    "api_key": api_key,
                    "rpm": 1000,
                }
            })
        elif openai_key:
            model_list.append({
                "model_name": "fast",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": openai_key,
                    "rpm": 1000,
                }
            })
        
        # Fallback
        if openai_key:
            model_list.append({
                "model_name": "fallback",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": openai_key,
                }
            })
        elif google_key:
            model_list.append({
                "model_name": "fallback",
                "litellm_params": {
                    "model": "gemini/gemini-2.0-flash",
                    "api_key": google_key,
                }
            })
        
        return model_list
    
    def _init_router(self) -> None:
        """Initialize LiteLLM router."""
        if not HAS_LITELLM:
            self._mock_mode = True
            return
        
        model_list = self._build_model_list()
        
        if not model_list:
            self._mock_mode = True
            return
        
        self._router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            num_retries=3,
            fallbacks=[{"default": ["fallback"]}] if any(m["model_name"] == "fallback" for m in model_list) else [],
            enable_pre_call_checks=True,
            cache_responses=True,
        )
    
    @property
    def chat(self):
        """Get chat interface (OpenAI-compatible)."""
        if self._router is None and not self._mock_mode:
            self._init_router()
        
        if self._mock_mode:
            return MockChatInterface()
        
        return self._router
    
    async def completion(
        self,
        messages: List[Dict],
        model: str = "default",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Any:
        """Generate a completion."""
        if self._router is None and not self._mock_mode:
            self._init_router()
        
        if self._mock_mode:
            return await MockChatInterface().completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
        
        return await self._router.acompletion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            **kwargs,
        )
    
    async def embed(self, text: str) -> List[float]:
        """Generate text embedding."""
        if self._mock_mode:
            # Return mock embedding
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            import random
            random.seed(hash_val)
            return [random.random() for _ in range(self.config.embedding_dimensions)]
        
        try:
            import litellm
            response = await litellm.aembedding(
                model=self.config.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception:
            # Fallback to mock
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            import random
            random.seed(hash_val)
            return [random.random() for _ in range(self.config.embedding_dimensions)]


class MockChatInterface:
    """Mock chat interface for testing without API keys."""
    
    def __init__(self):
        self.completions = MockCompletions()


class MockCompletions:
    """Mock completions interface."""
    
    async def create(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int = 1000,
        **kwargs,
    ) -> Any:
        """Generate mock completion."""
        # Extract last user message for context
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")[:100]
                break
        
        # Create mock response
        content = f"[Mock response to: {last_user}...]\n\nThis is a mock response for testing."
        
        return MockResponse(content)


class MockResponse:
    """Mock response object."""
    
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock choice object."""
    
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message object."""
    
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None
