"""Generate AI-powered practice questions for trading biases."""

from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys

from app.core.ai_config import get_ai_config
from app.core.logging import logger

# Thread pool for blocking LLM operations
_executor = ThreadPoolExecutor(max_workers=2)


class PracticeQuestion:
    """A single practice question with scenario, answers, and explanation."""

    def __init__(
        self,
        question_id: str,
        bias_type: str,
        scenario: str,
        answer_choices: list[str],
        correct_answer_index: int,
        explanation: str,
    ):
        self.question_id = question_id
        self.bias_type = bias_type
        self.scenario = scenario
        self.answer_choices = answer_choices
        self.correct_answer_index = correct_answer_index
        self.explanation = explanation

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "bias_type": self.bias_type,
            "scenario": self.scenario,
            "answer_choices": self.answer_choices,
            "correct_answer_index": self.correct_answer_index,
            "explanation": self.explanation,
        }


async def generate_practice_questions(
    top_two_biases: list[tuple[str, float]],  # [(bias_name, score), ...]
    trader_stats: Optional[dict] = None,
) -> list[dict]:
    """
    Generate 4 practice questions (2 per bias) using OpenAI.

    Args:
        top_two_biases: List of (bias_name, score) tuples, sorted by score descending
        trader_stats: Optional dict with trader statistics (win_rate, avg_trade_duration, etc)

    Returns:
        List of 4 practice questions, each with scenario, choices, correct answer, explanation
    """
    try:
        logger.info(f"📝 Starting practice questions generation for biases: {[b[0] for b in top_two_biases]}")
        ai_config = get_ai_config()

        # Debug: Check API key
        api_key = ai_config._get_api_key()
        logger.info(f"🔑 API key check - length: {len(api_key)}, loaded: {'YES' if api_key else 'NO'}")
        
        if not api_key:
            logger.error("❌ OPENAI_API_KEY is EMPTY - cannot call OpenAI")
            return _get_fallback_questions(top_two_biases)

        prompt = _build_practice_prompt(top_two_biases, trader_stats)

        # Initialize LLM
        logger.info("Initializing ChatOpenAI LLM...")
        llm = ai_config.get_llm()
        logger.info(f"LLM initialized: {llm.model_name}")

        # Call OpenAI in thread pool with timeout to avoid blocking event loop
        try:
            logger.info(f"Starting OpenAI API call (timeout: 2m)...")
            loop = asyncio.get_event_loop()
            # Run the blocking invoke in a thread pool to avoid blocking the event loop
            import time
            start_time = time.time()
            response = await asyncio.wait_for(
                loop.run_in_executor(_executor, lambda: llm.invoke(prompt)),
                timeout=120.0
            )
            elapsed = time.time() - start_time
            logger.info(f"OpenAI API call completed in {elapsed:.2f}s")
            logger.debug(f"OpenAI response type: {type(response)}")
            
        except asyncio.TimeoutError:
            logger.error("❌ OpenAI request timed out after 2 minutes. Using fallback questions.")
            return _get_fallback_questions(top_two_biases)

        questions = _parse_practice_response(response if isinstance(response, str) else response.content)
        logger.info(f"✓ Generated {len(questions)} practice questions from OpenAI")

        return questions

    except Exception as e:
        logger.error(f"❌ Error generating practice questions: {str(e)}", exc_info=True)
        # Fallback to hardcoded questions
        logger.info("Using fallback questions due to error")
        return _get_fallback_questions(top_two_biases)


def _build_practice_prompt(
    top_two_biases: list[tuple[str, float]],
    trader_stats: Optional[dict] = None,
) -> str:
    """Build the prompt for OpenAI to generate practice questions."""

    bias_descriptions = {
        "overtrading": "Making too many trades, especially in short bursts without proper breaks",
        "loss_aversion": "Taking losses too quickly while holding winners too long; fear of being wrong",
        "revenge_trading": "Entering aggressively after losses to 'get even' or prove the market wrong",
        "recency_bias": "Overweighting recent patterns and letting recent losses/wins dictate behavior",
    }

    bias_1_name, bias_1_score = top_two_biases[0]
    bias_2_name, bias_2_score = top_two_biases[1]

    bias_1_desc = bias_descriptions.get(bias_1_name, "a trading bias")
    bias_2_desc = bias_descriptions.get(bias_2_name, "a trading bias")

    prompt = f"""You are an expert trading coach creating a practice quiz to help a trader overcome their top two biases.

Trader's Top Two Biases:
1. {bias_1_name.replace('_', ' ').title()} (score: {bias_1_score:.0f}/100) - {bias_1_desc}
2. {bias_2_name.replace('_', ' ').title()} (score: {bias_2_score:.0f}/100) - {bias_2_desc}

Create exactly 4 practice questions: 2 questions for the first bias, 2 for the second bias.

For EACH question, provide:
1. A realistic trading scenario that matches the bias (describe what just happened, what the trader is feeling)
2. Four multiple-choice answer options (what the trader should do)
3. The index of the correct answer (0-3)
4. A plain-language explanation of why the correct answer is right, tied to this specific bias

Format your response as JSON with this exact structure:
{{
  "questions": [
    {{
      "bias": "{bias_1_name}",
      "scenario": "You just took a $500 loss on AAPL three minutes ago...",
      "choices": [
        "A) ...",
        "B) ...",
        "C) ...",
        "D) ..."
      ],
      "correct_index": 2,
      "explanation": "The right answer is C because..."
    }},
    ...
  ]
}}

Make scenarios vivid and emotionally realistic. Make answer choices tempting but include one clear best choice.
Explanations should reference the specific bias and suggest what the trader should do instead."""

    return prompt


def _parse_practice_response(response_text: str) -> list[dict]:
    """Parse OpenAI response JSON into practice questions."""
    import json

    try:
        # Extract JSON from response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")

        json_str = response_text[start:end]
        data = json.loads(json_str)

        questions = []
        for i, q in enumerate(data.get("questions", [])):
            questions.append(
                {
                    "question_id": f"q{i+1}",
                    "bias": q.get("bias", "unknown"),
                    "scenario": q.get("scenario", ""),
                    "choices": q.get("choices", []),
                    "correct_index": q.get("correct_index", 0),
                    "explanation": q.get("explanation", ""),
                }
            )

        return questions[:4]  # Return max 4 questions

    except Exception as e:
        logger.error(f"Error parsing practice response: {str(e)}")
        return []


def _get_fallback_questions(top_two_biases: list[tuple[str, float]]) -> list[dict]:
    """Return hardcoded fallback practice questions."""

    bias_scenarios = {
        "overtrading": [
            {
                "scenario": "It's 10:15 AM. You've made 8 trades already this morning, all in the first 45 minutes. Your account is up $200. You see a quick scalp setup on TSLA. Your hand is itching to click the order button.",
                "choices": [
                    "Execute the trade - you're hot and the setup looks clean",
                    "Take a 30-minute break and just watch the market",
                    "Place a smaller position (half size) and see if you can stay disciplined",
                    "Post a screenshot of your equity curve on trading Discord",
                ],
                "correct_index": 1,
                "explanation": "The overtrading bias makes frequent trading feel rewarding even when risk-adjusted returns are poor. After 8 trades in 45 minutes, you need a break. Your edges fade when you repeat them constantly. The best action is to step back and reset.",
            },
            {
                "scenario": "You've closed 6 winners in a row, each held for 2-4 minutes. Your dopamine is PUMPING. Trade 7 sets up the same way. You're about to go all-in with max size.",
                "choices": [
                    "Max out position size - you're in a hot streak",
                    "Take the trade at normal size, let it play out",
                    "Stop trading for the rest of the day - streaks end",
                    "Add to winners as they go deeper in the money",
                ],
                "correct_index": 2,
                "explanation": "Streaks are often just randomness or survivorship bias. When you feel 'hot,' your risk management usually gets worse, not better. Ending the day early when up is a key rule to protect overtrading traders.",
            },
        ],
        "loss_aversion": [
            {
                "scenario": "You're down $1,500 on NVDA. The setup still looks good but you're uncomfortable. You're staring at the loss on your screen. Your profit target is still 4 cents away but you want to get out NOW.",
                "choices": [
                    "Exit immediately and take the loss - stop the pain",
                    "Stay in the trade and scale down position size",
                    "Move your stop loss down to lock in a smaller loss",
                    "Hold your full position and follow your original plan",
                ],
                "correct_index": 3,
                "explanation": "Loss aversion makes losses feel 2-3x worse than gains feel good. This biases you to exit winnable trades too early to avoid the pain. Your plan had a reason - stick to it unless the *setup* changes, not your emotions.",
            },
            {
                "scenario": "You're up $800 on SPY. You've been holding for 20 minutes. Now the price is retracing 2 cents from the high. Your instinct is to take the profit NOW before it disappears.",
                "choices": [
                    "Exit and take the nice profit",
                    "Move your stop loss up to breakeven",
                    "Stay in and don't move your stop - let winners run",
                    "Average up to maximize gains",
                ],
                "correct_index": 2,
                "explanation": "Loss-averse traders hold onto losses and exit winners too fast. Small retracements are normal. If you're in a valid uptrend, don't exit just to capture profit early. Let your winners run to make up for the losses you couldn't avoid.",
            },
        ],
        "revenge_trading": [
            {
                "scenario": "You just got stopped out on QQQ for a $600 loss. You're feeling stupid. You see another setup on QQQ 3 minutes later. You're thinking 'I'll get that $600 back right now on a bigger size.'",
                "choices": [
                    "Take the trade at 2x normal size to recover the loss faster",
                    "Take just ONE more trade at normal size - it's a good setup",
                    "Skip this trade. Set a timer for 15 minutes before you trade again",
                    "Take the trade at 1.5x size to stay aggressive",
                ],
                "correct_index": 2,
                "explanation": "Revenge trading is entering too soon after a loss with bad risk management. Your judgment is clouded by emotion. When you feel the urge to 'get it back,' that's exactly when you should NOT trade. A 15-minute timeout resets your mind.",
            },
            {
                "scenario": "You took a $400 loss. Now you're eyeing the SAME STOCK. Your chart says 'avoid for 2 hours' due to your rules, but you have a 'revenge feeling' - like you should re-enter to prove you were right.",
                "choices": [
                    "Re-enter the same stock immediately at 1.5x size",
                    "Re-enter at normal size - it's still a valid setup",
                    "Skip this. Respect your 2-hour cooldown rule and trade something else",
                    "Wait 30 minutes then re-enter the same stock",
                ],
                "correct_index": 2,
                "explanation": "Your cooldown rule exists because revenge traders have bad judgment after losses. Revenge trades average a loss of 40% more than normal trades. Discipline is boring but profitable.",
            },
        ],
        "recency_bias": [
            {
                "scenario": "Your last 3 trades lost money. You're seeing setups everywhere but you're suddenly TERRIFIED to take them. You're thinking 'Maybe I've lost my edge. Maybe I should just sit out for a while.'",
                "choices": [
                    "Sit out for the rest of the week to recharge",
                    "Reduce position size to 50%",
                    "Take your next three setups at normal size (stick to your process)",
                    "Close your account - you're not cut out for this",
                ],
                "correct_index": 2,
                "explanation": "Recency bias makes a 3-trade losing streak feel like you've lost your edge forever. It hasn't. Your edge is long-term. Three losses is noise. Trust your process, not your emotions about recent results.",
            },
            {
                "scenario": "You're up $5,000 on the week. Your last 4 trades were winners. Now you see a marginal setup. You're thinking 'My edge is BACK. Let me size up and ride this hot streak.'",
                "choices": [
                    "Increase position size to 1.5x - you're hitting your stride",
                    "Take the trade at 2x size - ride the hot streak",
                    "Take the trade at your normal size and follow your risk rules",
                    "Skip this setup - marginal setups aren't worth it after a streak",
                ],
                "correct_index": 2,
                "explanation": "A 4-trade win streak is not signal that your edge suddenly got better. Recency bias makes short-term results feel permanent. Your size, your rules, your process - none of these change because of this week's results.",
            },
        ],
    }

    questions = []
    question_id = 1

    for bias_name, _ in top_two_biases[:2]:
        bias_questions = bias_scenarios.get(bias_name, [])[:2]

        for q in bias_questions:
            questions.append(
                {
                    "question_id": f"q{question_id}",
                    "bias": bias_name,
                    "scenario": q["scenario"],
                    "choices": q["choices"],
                    "correct_index": q["correct_index"],
                    "explanation": q["explanation"],
                }
            )
            question_id += 1

    return questions[:4]
