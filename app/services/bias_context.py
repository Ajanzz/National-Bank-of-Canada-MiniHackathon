"""Generate market context explanations for detected trading biases."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.core.ai_config import get_ai_config
from app.core.logging import logger

# Thread pool for blocking LLM operations
_executor = ThreadPoolExecutor(max_workers=2)


async def get_bias_context_explanations(
    top_two_biases: list[dict],  # [{"bias_type": "overtrading", "score": 85}, ...]
    date_range: Optional[dict] = None,
    num_events: int = 5,
) -> list[dict]:
    """
    Generate market context explanations for trader's top 2 biases.
    
    Returns structured explanations with:
    - Market events from trading period
    - Connection to detected biases
    - Practical takeaways
    - Personal reasons placeholder
    """
    try:
        logger.info(f"📊 Generating bias context explanations for biases: {[b['bias_type'] for b in top_two_biases]}")
        
        ai_config = get_ai_config()
        
        if not ai_config._get_api_key() or len(ai_config._get_api_key()) <= 20:
            logger.warning("OpenAI API key not configured. Using fallback context.")
            return _get_fallback_context(top_two_biases, date_range)
        
        prompt = _build_context_prompt(top_two_biases, date_range, num_events)
        
        # Initialize LLM
        logger.info("Initializing ChatOpenAI LLM for bias context...")
        llm = ai_config.get_llm()
        
        # Call OpenAI in thread pool with 2-minute timeout
        try:
            logger.info(f"Starting OpenAI API call for bias context (timeout: 2m)...")
            loop = asyncio.get_event_loop()
            import time
            start_time = time.time()
            response = await asyncio.wait_for(
                loop.run_in_executor(_executor, lambda: llm.invoke(prompt)),
                timeout=120.0
            )
            elapsed = time.time() - start_time
            logger.info(f"OpenAI API call completed in {elapsed:.2f}s")
            
        except asyncio.TimeoutError:
            logger.error("❌ OpenAI request timed out after 2 minutes. Using fallback context.")
            return _get_fallback_context(top_two_biases, date_range)
        
        explanations = _parse_context_response(response.content if hasattr(response, 'content') else response)
        logger.info(f"✓ Generated {len(explanations)} bias context explanations from OpenAI")
        
        return explanations
    
    except Exception as e:
        logger.error(f"Error generating bias context: {str(e)}", exc_info=True)
        # Fallback to static explanations
        return _get_fallback_context(top_two_biases, date_range)


def _build_context_prompt(
    top_two_biases: list[dict],
    date_range: Optional[dict],
    num_events: int,
) -> str:
    """Build the prompt for OpenAI to generate bias context."""
    
    bias_info = "\n".join([
        f"- {b['bias_type'].replace('_', ' ').title()} (score: {b.get('score', 0)}/100)"
        for b in top_two_biases
    ])
    
    prompt = f"""You are a financial analyst explaining market context for a trader's behavioral biases.

Trader's Detected Biases (traded today):
{bias_info}

Your task: For EACH bias, provide:
1. A list of 3-5 real, recent market/economic events from this week or recent past that could emotionally trigger or explain why this bias pattern appears in traders
2. A plain-language explanation (2-3 sentences) connecting those recent events to WHY traders experience this bias
3. ONE practical, specific action the trader can take to reduce this bias during similar market conditions

Format your response as JSON with this exact structure:
{{
  "explanations": [
    {{
      "bias_name": "Overtrading",
      "market_events": [
        {{"date": "Jan 15", "headline": "Market volatility surge - VIX jumps 20%"}},
        {{"date": "Jan 18", "headline": "Tech earnings beat expectations"}},
        {{"date": "Jan 22", "headline": "Rapid intraday reversals in major indices"}}
      ],
      "connection_explanation": "During volatile periods with high news flow and rapid price swings, traders feel pressure to respond constantly. The fear of missing opportunities can trigger overtrading as you try to keep up with market action.",
      "practical_takeaway": "When market volatility spikes above normal, enforce a max trades per hour rule (e.g., 4 trades/hr). Use a timer to enforce breaks."
    }},
    ...
  ]
}}

Guidelines:
- Use ONLY real, recent, publicly documented market events
- Focus on emotional/psychological triggers, not just statistics
- Make the connection clear: what about these events would emotionally trigger THIS bias?
- Practical takeaways must be specific and immediately actionable
- You can use approximate dates (Jan 15, etc.) for recent events"""

    return prompt


def _parse_context_response(response_text: str) -> list[dict]:
    """Parse OpenAI response into structured bias context explanations."""
    import json
    
    try:
        # Extract JSON from response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        
        json_str = response_text[start:end]
        data = json.loads(json_str)
        
        return data.get("explanations", [])
    
    except Exception as e:
        logger.error(f"Error parsing context response: {str(e)}")
        return []


def _get_fallback_context(
    top_two_biases: list[dict],
    date_range: Optional[dict],
) -> list[dict]:
    """Return hardcoded fallback bias context explanations."""
    
    fallback_contexts = {
        "overtrading": {
            "bias_name": "Overtrading",
            "market_events": [
                {"date": "2024-01-15", "headline": "Market volatility spike - VIX jumps 20%"},
                {"date": "2024-01-18", "headline": "Earnings season begins with mixed results"},
                {"date": "2024-01-22", "headline": "Rapid reversals in tech stocks - intraday swings > 5%"},
            ],
            "connection_explanation": "During volatile periods with rapid price moves, traders often feel pressure to respond constantly. The fear of missing a big move can trigger overtrading as you try to 'keep up' with market action.",
            "practical_takeaway": "Set a max trades per hour limit (e.g., 4 trades/hr). When you hit it, step away for 15 minutes. Use an alarm or app to enforce this.",
        },
        "loss_aversion": {
            "bias_name": "Loss Aversion",
            "market_events": [
                {"date": "2024-01-10", "headline": "Unexpected economic data - inflation higher than expected"},
                {"date": "2024-01-14", "headline": "Fed official hints at longer rate hold - market drops 2%"},
                {"date": "2024-01-20", "headline": "Correlation breakdown between traditional hedges"},
            ],
            "connection_explanation": "When market conditions become unpredictable and uncertainty rises, traders become more reluctant to hold losers. The feeling of losing control makes losses feel more painful.",
            "practical_takeaway": "Before each trade, write down your 3 reasons for entering AND a specific exit price. Stick to it regardless of how the trade 'feels'.",
        },
        "revenge_trading": {
            "bias_name": "Revenge Trading",
            "market_events": [
                {"date": "2024-01-12", "headline": "Gap down open catches traders off guard"},
                {"date": "2024-01-16", "headline": "Flash crash in index futures - brief but sharp"},
                {"date": "2024-01-19", "headline": "Recovery just as fast - whipsaw trades"},
            ],
            "connection_explanation": "After sudden, sharp losses, especially in chaotic market conditions, traders feel the urge to 'get even' quickly. The sudden loss triggers emotional urgency.",
            "practical_takeaway": "Create a 30-minute post-loss lockout rule. After any loss >= 2x your target profit, you must wait 30 min before trading again.",
        },
        "recency_bias": {
            "bias_name": "Recency Bias",
            "market_events": [
                {"date": "2024-01-08", "headline": "Three days of consecutive gains - momentum rally"},
                {"date": "2024-01-11", "headline": "Breakout above key resistance"},
                {"date": "2024-01-17", "headline": "Sudden reversal - momentum fizzles"},
            ],
            "connection_explanation": "After a strong winning streak, traders overweight recent wins and expect them to continue. When the pattern suddenly reverses, the trader is caught off-guard.",
            "practical_takeaway": "Review your last 20 trades every Friday, not just yesterday's results. Compare today's setup to stats from the last 3 months, not just the last 3 days.",
        },
    }
    
    explanations = []
    for bias in top_two_biases[:2]:
        bias_type = bias.get("bias_type", "").lower()
        
        # Try to find matching fallback
        for key, context in fallback_contexts.items():
            if key in bias_type or bias_type in key:
                explanations.append(context)
                break
        
        # If no match, use a generic one
        if len(explanations) <= len(top_two_biases) - 1:
            explanations.append({
                "bias_name": bias.get("bias_type", "Unknown Bias").replace("_", " ").title(),
                "market_events": [
                    {"date": "2024-01-15", "headline": "Market volatility and uncertainty increased"},
                    {"date": "2024-01-20", "headline": "Trading conditions shifted unexpectedly"},
                ],
                "connection_explanation": "Market conditions and rapid price movements can trigger emotional trading behaviors.",
                "practical_takeaway": "Review your trading plan before market open. Stick to your rules even when emotions run high.",
            })
    
    return explanations[:2]
