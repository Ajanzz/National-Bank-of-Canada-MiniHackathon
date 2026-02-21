"""LangChain-based AI services for trading insights and coaching."""

from typing import Optional

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from app.core.ai_config import ai_config
from app.core.logging import logger


class TradingCoachService:
    """Service for AI-powered trading coaching via LangChain."""

    def __init__(self):
        self.memory: Optional[ConversationBufferMemory] = None
        self.chain: Optional[LLMChain] = None

    def initialize(self) -> bool:
        """Initialize the coach with LangChain."""
        try:
            if not ai_config.validate():
                logger.error("AI config validation failed")
                return False

            self.memory = ai_config.create_memory()
            prompt = ai_config.create_trading_coach_prompt()
            llm = ai_config.get_llm()

            self.chain = LLMChain(
                llm=llm,
                prompt=prompt,
                memory=self.memory,
                verbose=False,
            )
            logger.info("TradingCoachService initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TradingCoachService: {str(e)}")
            return False

    async def chat(
        self,
        user_query: str,
        trading_context: str = "",
    ) -> Optional[str]:
        """
        Chat with the trading coach.

        Args:
            user_query: User's question or message
            trading_context: Relevant trading data/context (optional)

        Returns:
            Coach response or None if not initialized
        """
        if not self.chain or not self.memory:
            logger.error("TradingCoachService not initialized")
            return None

        try:
            response = self.chain.invoke(
                {
                    "user_query": user_query,
                    "trading_context": trading_context,
                },
            )
            return response.get("text", "")
        except Exception as e:
            logger.error(f"Error in coach chat: {str(e)}")
            return None

    def get_chat_history(self) -> list:
        """Get conversation history."""
        if not self.memory:
            return []
        return self.memory.chat_memory.messages if hasattr(self.memory, "chat_memory") else []

    def clear_history(self) -> None:
        """Clear conversation history."""
        if self.memory:
            self.memory.clear()


# Global coach instance
trading_coach = TradingCoachService()
