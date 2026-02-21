"""Educational finance coach chatbot service using OpenAI."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.core.ai_config import get_ai_config
from app.core.logging import logger

# Thread pool for blocking LLM operations
_executor = ThreadPoolExecutor(max_workers=2)


async def get_coach_response(
    chat_history: list[dict],  # [{"role": "user"/"assistant", "content": "..."}, ...]
    reflection_notes: Optional[list[str]] = None,
) -> str:
    """
    Generate an educational finance coach response.
    
    Uses conversation history to provide context-aware answers.
    Focuses on concepts, risks, and general approaches.
    Avoids direct buy/sell recommendations.
    """
    try:
        logger.info(f"🎓 Coach generating response from {len(chat_history)} messages")
        
        ai_config = get_ai_config()
        
        if not ai_config._get_api_key() or len(ai_config._get_api_key()) <= 20:
            logger.warning("OpenAI API key not configured. Using fallback response.")
            return _get_fallback_response(chat_history)
        
        # Build system prompt
        system_prompt = _build_system_prompt()
        
        # Convert chat history to LangChain format - handle both dict and Pydantic models
        messages = []
        for msg in chat_history:
            if isinstance(msg, dict):
                messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                # Pydantic model
                messages.append({"role": msg.role, "content": msg.content})
        
        # Initialize LLM
        logger.info("Initializing ChatOpenAI LLM for coach chat...")
        llm = ai_config.get_llm()
        
        # Call OpenAI in thread pool with 30-second timeout
        try:
            logger.info(f"Starting OpenAI API call for coach chat (timeout: 30s)...")
            loop = asyncio.get_event_loop()
            import time
            start_time = time.time()
            
            # Build the full message with system prompt
            full_message = f"{system_prompt}\n\nConversation:\n"
            for msg in messages:
                if msg["role"] == "user":
                    full_message += f"User: {msg['content']}\n"
                else:
                    full_message += f"Coach: {msg['content']}\n"
            full_message += "Coach:"
            
            response = await asyncio.wait_for(
                loop.run_in_executor(_executor, lambda: llm.invoke(full_message)),
                timeout=30.0
            )
            elapsed = time.time() - start_time
            logger.info(f"OpenAI API call completed in {elapsed:.2f}s")
            
        except asyncio.TimeoutError:
            logger.error("❌ OpenAI request timed out after 30 seconds. Using fallback response.")
            return _get_fallback_response(chat_history)
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"✓ Generated coach response: {len(response_text)} chars")
        
        # Clean up the response (remove "Coach:" prefix if it was added)
        response_text = response_text.strip()
        if response_text.startswith("Coach:"):
            response_text = response_text[6:].strip()
        
        return response_text
    
    except Exception as e:
        logger.error(f"Error generating coach response: {str(e)}", exc_info=True)
        return _get_fallback_response(chat_history)


def _build_system_prompt() -> str:
    """Build the system prompt for the finance coach."""
    return """You are an educational finance coach. Your role is to:

1. EDUCATE about financial concepts, trading strategies, and risk management
2. EXPLAIN how markets work, what tools traders use, and general best practices
3. DISCUSS risks and potential downsides of different approaches
4. SHARE general principles and frameworks for decision-making

IMPORTANT GUIDELINES:
- NEVER give specific buy/sell recommendations (e.g., "buy TSLA", "sell at $150")
- NEVER provide personalized financial advice for the user's specific situation
- NEVER guarantee any outcomes or suggest you know what the market will do
- ALWAYS remind users these are educational concepts, not personalized advice
- ALWAYS focus on frameworks, principles, and risk awareness
- Use plain language, avoid jargon when possible
- Keep responses concise (2-3 sentences per point)
- If asked for specific trading advice, politely redirect to general principles

Example topics you can help with:
- What is a stop-loss and why traders use them?
- How does diversification reduce risk?
- Explain what technical analysis is and its limitations
- What are the different order types in stock trading?
- How do traders manage psychological biases?
- What is volatility and how does it affect trading?
- General risk management principles

Tone: Helpful, educational, professional, and cautious about giving specific advice."""


def _get_fallback_response(chat_history: list[dict]) -> str:
    """Return educational fallback response when API is unavailable."""
    
    # Get the last user message to provide context-aware fallback
    last_user_msg = None
    for msg in reversed(chat_history):
        # Handle both dict and Pydantic model objects
        if isinstance(msg, dict):
            role = msg["role"]
            content = msg["content"]
        else:
            role = msg.role
            content = msg.content
        
        if role == "user":
            last_user_msg = content.lower()
            break
    
    # Map user questions to fallback educational responses
    fallback_responses = {
        "cpi": "CPI (Consumer Price Index) measures inflation by tracking price changes in a basket of consumer goods and services over time. Higher inflation can affect markets, interest rates, and purchasing power. It's important for traders to understand that inflation data is one economic indicator among many that influences market movements.",
        
        "stop-loss": "A stop-loss is an order that automatically sells a security if it reaches a certain price below your entry point. It helps limit losses on a trade. For example, if you buy at $100 and set a stop-loss at $95, your position will sell if the price drops to $95. This is a risk management tool, not a guarantee of execution price during fast markets.",
        
        "diversification": "Diversification means spreading investments across different assets, sectors, or strategies so that no single loss significantly impacts overall performance. The idea: if one holding drops, others may stay stable or gain. However, diversification doesn't eliminate risk—all investments can lose value in certain market conditions.",
        
        "limit order": "A limit order is an instruction to buy or sell a security at a specific price or better. For example, a buy limit at $50 means 'only buy if the price reaches $50 or lower.' Limit orders don't guarantee execution if the price never reaches your limit, but they give you price control.",
    }
    
    # Find best matching response
    for keyword, response in fallback_responses.items():
        if last_user_msg and keyword in last_user_msg:
            return response
    
    # Default response
    return "I'm currently in fallback mode and can't access live AI responses. Feel free to ask about trading concepts like stop-losses, diversification, limit orders, or other trading topics. I'm here to provide educational information, though I can't give personalized financial advice."


