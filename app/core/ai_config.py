"""LangChain and OpenAI configuration."""

import os
from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI

from app.core.logging import logger

# Try to import optional LangChain prompt components.
# Support both current (langchain_core) and legacy module paths.
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    try:
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError:
        ChatPromptTemplate = None
        MessagesPlaceholder = None
        logger.warning("ChatPromptTemplate/MessagesPlaceholder not available.")

# Conversation memory is optional and may require an extra package depending on LangChain version.
try:
    from langchain_classic.memory import ConversationBufferMemory
except ImportError:
    try:
        from langchain.memory import ConversationBufferMemory
    except ImportError:
        ConversationBufferMemory = None


class AIConfig:
    """Centralized AI/LangChain configuration."""

    def __init__(self):
        self.temperature = 0.7
        self.max_tokens = 2000
        self._llm_cache = None  # Cache LLM instance

    def _get_api_key(self) -> str:
        """Get API key from environment (read on-demand, never cached)."""
        key = os.getenv("OPENAI_API_KEY", "")
        logger.debug(f"_get_api_key called: length={len(key)}, first 20 chars={key[:20] if key else 'EMPTY'}")
        return key

    def _get_model(self) -> str:
        """Get model from environment."""
        return os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

    def validate(self) -> bool:
        """Validate API key is configured."""
        api_key = self._get_api_key()
        logger.info(f"validate(): api_key length={len(api_key)}")

        if not api_key:
            logger.error("[ERROR] API key is EMPTY")
            return False

        if api_key == "sk-your-api-key-here":
            logger.error("[ERROR] API key is PLACEHOLDER value")
            return False

        if not api_key.startswith("sk-"):
            logger.error(f"[ERROR] API key has invalid format, starts with: {api_key[:10]}")
            return False

        logger.info(f"[OK] API key validated (length: {len(api_key)})")
        return True

    def get_llm(self):
        """Get cached ChatOpenAI instance."""
        if self._llm_cache is not None:
            logger.debug("Returning cached LLM instance")
            return self._llm_cache

        logger.info("Initializing ChatOpenAI LLM instance...")
        if not self.validate():
            raise ValueError("OpenAI API key not configured")

        logger.info(f"Creating ChatOpenAI with API key length: {len(self._get_api_key())}")
        self._llm_cache = ChatOpenAI(
            api_key=self._get_api_key(),
            model=self._get_model(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        logger.info(f"[OK] ChatOpenAI LLM initialized (model: {self._get_model()})")
        return self._llm_cache

    @staticmethod
    def create_memory() -> Optional[object]:
        """Create conversation memory for chat interactions."""
        if ConversationBufferMemory is None:
            logger.warning("ConversationBufferMemory not available")
            return None

        return ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
        )

    @staticmethod
    def create_trading_coach_prompt() -> Optional[object]:
        """Create prompt template for trading coach."""
        if ChatPromptTemplate is None or MessagesPlaceholder is None:
            logger.warning("ChatPromptTemplate/MessagesPlaceholder not available")
            return None

        return ChatPromptTemplate(
            input_variables=["user_query", "trading_context"],
            messages=[
                (
                    "system",
                    """You are an expert trading coach specializing in helping traders recognize and overcome behavioral biases.

Your role is to:
1. Analyze trading decisions and identify potential biases
2. Provide constructive, non-judgmental coaching
3. Suggest practical improvements based on trading psychology principles
4. Help traders develop discipline and better decision-making habits

When analyzing trades:
- Look for patterns of emotional decision-making
- Identify repeating mistakes
- Suggest concrete rules to prevent biased trades
- Be empathetic but direct about areas for improvement

Keep responses concise and actionable.""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_query}\n\nTrading Context:\n{trading_context}"),
            ],
        )


# Global AI config instance
ai_config = AIConfig()


def verify_openai_config() -> bool:
    """Verify OpenAI API key is properly configured at startup."""
    try:
        from app.core.config import get_settings

        settings = get_settings()
        api_key = settings.openai_api_key

        logger.info(f"DEBUG: verify_openai_config - api_key length: {len(api_key)}")

        if not api_key:
            logger.error("[ERROR] OPENAI_API_KEY is EMPTY")
            logger.error("Please add OPENAI_API_KEY to your .env file")
            return False

        if api_key == "sk-your-api-key-here":
            logger.error("[ERROR] OPENAI_API_KEY has placeholder value")
            logger.error("Please update OPENAI_API_KEY in your .env file with your actual API key")
            return False

        if not api_key.startswith("sk-"):
            logger.error("[ERROR] OPENAI_API_KEY has invalid format")
            return False

        logger.info(f"[OK] OpenAI API key detected (length: {len(api_key)}, starts: {api_key[:10]})")
        return True
    except Exception as e:
        logger.error(f"Error verifying OpenAI config: {e}")
        return False


@lru_cache(maxsize=1)
def get_ai_config() -> AIConfig:
    """Get cached AI config instance."""
    return ai_config
