"""Generate bias context and emotional check-in personalization."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from app.core.ai_config import get_ai_config
from app.core.logging import logger

BiasType = Literal["REVENGE_TRADING", "LOSS_AVERSION", "RECENCY_BIAS", "OVERTRADING"]
YesNoAnswer = Optional[Literal["YES", "NO"]]

BIAS_DISPLAY_NAMES: dict[BiasType, str] = {
    "REVENGE_TRADING": "Revenge Trading",
    "LOSS_AVERSION": "Loss Aversion",
    "RECENCY_BIAS": "Recency Bias",
    "OVERTRADING": "Overtrading",
}

BIAS_LOOKUP: dict[str, BiasType] = {
    "REVENGE_TRADING": "REVENGE_TRADING",
    "revenge_trading": "REVENGE_TRADING",
    "Revenge Trading": "REVENGE_TRADING",
    "LOSS_AVERSION": "LOSS_AVERSION",
    "loss_aversion": "LOSS_AVERSION",
    "Loss Aversion": "LOSS_AVERSION",
    "RECENCY_BIAS": "RECENCY_BIAS",
    "recency_bias": "RECENCY_BIAS",
    "Recency Bias": "RECENCY_BIAS",
    "OVERTRADING": "OVERTRADING",
    "overtrading": "OVERTRADING",
    "Overtrading": "OVERTRADING",
}

# Hardcoded emotional check-in question bank (must remain static)
EMOTIONAL_CHECKIN_QUESTIONS: dict[BiasType, list[str]] = {
    "REVENGE_TRADING": [
        "After a loss, do you feel a strong urge to \u2018win it back\u2019 right away?",
        "When you feel frustrated or embarrassed by a trade, is it harder to pause before taking the next one?",
        "Do you tend to increase risk when you feel like you\u2019re \u2018behind\u2019 for the day?",
    ],
    "LOSS_AVERSION": [
        "Do you avoid closing a losing trade because accepting the loss feels emotionally painful?",
        "When a trade goes against you, do you keep holding mainly because you hope it will turn around?",
        "Do you feel more regret from taking a loss than satisfaction from taking a similar-sized gain?",
    ],
    "RECENCY_BIAS": [
        "After a recent win or loss streak, do you feel unusually confident or unusually doubtful about the next trade?",
        "Do your last few trades strongly influence how you size positions right now?",
        "When the market just moved, do you feel pressure to act quickly so you don\u2019t miss out?",
    ],
    "OVERTRADING": [
        "When you\u2019re bored, anxious, or restless, do you trade just to feel engaged?",
        "Do you feel uneasy when you\u2019re not in a trade, like you\u2019re missing opportunities?",
        "On stressful days, do you notice you take more trades than you planned, even without clear setups?",
    ],
}
FALLBACK_HABITS: dict[BiasType, list[str]] = {
    "REVENGE_TRADING": [
        "Use a fixed cool-down timer after any loss before reviewing the next setup.",
        "Write one neutral sentence about what happened before placing another order.",
        "Return to your normal position size only after you can restate your entry criteria clearly.",
    ],
    "LOSS_AVERSION": [
        "Define your exit condition before entry and keep it visible during the trade.",
        "Name the emotion you feel when a position is red, then re-check your written plan.",
        "Use a brief post-trade note to separate process quality from outcome.",
    ],
    "RECENCY_BIAS": [
        "Score each setup against a checklist before placing orders, independent of recent streaks.",
        "Review a larger sample of prior trades once daily before market open.",
        "Set a 60-second pause rule after sharp moves before acting.",
    ],
    "OVERTRADING": [
        "Set planned trade-count limits and take a scheduled break after each block.",
        "Do a short body check (breath, tension, urgency) before every order.",
        "If you feel restless, step away for two minutes before evaluating a new setup.",
    ],
}

FALLBACK_CONNECTIONS: dict[BiasType, str] = {
    "REVENGE_TRADING": (
        "Sharp losses can trigger urgency and self-pressure. That emotional surge can reduce patience "
        "and make the next decision feel like a recovery mission instead of a planned trade."
    ),
    "LOSS_AVERSION": (
        "When losses feel personally painful, it is natural to hesitate on exits. That can blur the line "
        "between a planned hold and an emotion-driven hold."
    ),
    "RECENCY_BIAS": (
        "Recent wins or losses can feel more important than longer-term data. This can shift confidence too far "
        "in either direction and distort position sizing."
    ),
    "OVERTRADING": (
        "Periods of stress, boredom, or high stimulation can create pressure to stay active. That pressure can "
        "lead to extra trades outside your planned setup quality."
    ),
}

# Thread pool for blocking LLM operations
_executor = ThreadPoolExecutor(max_workers=2)


def normalize_bias_type(raw_bias_type: str) -> Optional[BiasType]:
    """Normalize frontend/backend bias labels to canonical bias enum."""
    return BIAS_LOOKUP.get(raw_bias_type)


def _normalize_top_two_biases(top_two_biases: list[dict]) -> list[dict]:
    """Keep only recognized biases and return the top 2 by score."""
    best_score_by_bias: dict[BiasType, int] = {}
    for item in top_two_biases:
        raw = str(item.get("bias_type", ""))
        bias_type = normalize_bias_type(raw)
        if not bias_type:
            continue

        try:
            score = int(round(float(item.get("score", 0))))
        except (TypeError, ValueError):
            score = 0

        bounded_score = max(0, min(100, score))
        current_best = best_score_by_bias.get(bias_type)
        if current_best is None or bounded_score > current_best:
            best_score_by_bias[bias_type] = bounded_score

    normalized = [
        {
            "bias_type": bias_type,
            "bias_name": BIAS_DISPLAY_NAMES[bias_type],
            "score": score,
        }
        for bias_type, score in best_score_by_bias.items()
    ]
    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized[:2]


def _normalize_checkin_payload(emotional_check_in: Optional[list[dict]]) -> dict[BiasType, list[dict]]:
    """Normalize check-in responses keyed by canonical bias type.

    Each selected bias is normalized to the fixed 3-question bank with
    YES/NO/Skip (null) answers only.
    """
    by_bias: dict[BiasType, list[dict]] = {}
    if not emotional_check_in:
        return by_bias

    for section in emotional_check_in:
        raw_bias = str(section.get("bias_type", ""))
        bias_type = normalize_bias_type(raw_bias)
        if not bias_type:
            continue

        raw_responses = section.get("responses", [])
        if not isinstance(raw_responses, list):
            continue

        answers_by_index: dict[int, YesNoAnswer] = {0: None, 1: None, 2: None}
        for idx, item in enumerate(raw_responses[:3]):
            answer_raw = item.get("answer")
            answers_by_index[idx] = answer_raw if answer_raw in ("YES", "NO") else None

        normalized_responses = [
            {
                "question_id": f"{bias_type}_Q{idx + 1}",
                "question": EMOTIONAL_CHECKIN_QUESTIONS[bias_type][idx],
                "answer": answers_by_index[idx],
            }
            for idx in range(3)
        ]

        by_bias[bias_type] = normalized_responses
    return by_bias


def normalize_top_two_biases_input(top_two_biases: list[dict]) -> list[dict]:
    """Public wrapper so API routes can reuse canonical top-2 normalization."""
    return _normalize_top_two_biases(top_two_biases)


def normalize_emotional_checkin_input(emotional_check_in: Optional[list[dict]]) -> dict[BiasType, list[dict]]:
    """Public wrapper so API routes can reuse canonical check-in normalization."""
    return _normalize_checkin_payload(emotional_check_in)


def _build_checkin_summary(responses: list[dict]) -> dict[str, int]:
    yes_count = sum(1 for r in responses if r.get("answer") == "YES")
    no_count = sum(1 for r in responses if r.get("answer") == "NO")
    skipped_count = sum(1 for r in responses if r.get("answer") is None)
    return {"yes_count": yes_count, "no_count": no_count, "skipped_count": skipped_count}


async def get_bias_context_explanations(
    top_two_biases: list[dict],
    emotional_check_in: Optional[list[dict]] = None,
    date_range: Optional[dict] = None,
    num_events: int = 5,
) -> dict:
    """
    Generate market context + personalized emotional context for top two biases.

    Input data includes top bias scores and the user's YES/NO/Skip answers to fixed
    emotional check-in questions. Output is safe, educational, and process-focused.
    """
    normalized_biases = _normalize_top_two_biases(top_two_biases)
    normalized_checkins = _normalize_checkin_payload(emotional_check_in)

    try:
        logger.info(
            "Generating bias context for biases=%s",
            [b["bias_type"] for b in normalized_biases],
        )

        ai_config = get_ai_config()
        api_key = ai_config._get_api_key()
        if not api_key or len(api_key) <= 20:
            logger.warning("OpenAI API key not configured. Using fallback bias context.")
            return _get_fallback_context(normalized_biases, normalized_checkins, date_range)

        prompt = _build_context_prompt(
            top_two_biases=normalized_biases,
            normalized_checkins=normalized_checkins,
            date_range=date_range,
            num_events=num_events,
        )
        llm = ai_config.get_llm()

        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(_executor, lambda: llm.invoke(prompt)),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            logger.error("OpenAI request timed out for bias context. Using fallback.")
            return _get_fallback_context(normalized_biases, normalized_checkins, date_range)

        response_text = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_context_response(response_text)
        if not parsed:
            logger.warning("Could not parse bias context response. Using fallback.")
            return _get_fallback_context(normalized_biases, normalized_checkins, date_range)

        return _merge_with_fallback_defaults(parsed, normalized_biases, normalized_checkins, date_range)
    except Exception as exc:
        logger.error(f"Error generating bias context: {exc}", exc_info=True)
        return _get_fallback_context(normalized_biases, normalized_checkins, date_range)


def _build_context_prompt(
    top_two_biases: list[dict],
    normalized_checkins: dict[BiasType, list[dict]],
    date_range: Optional[dict],
    num_events: int,
) -> str:
    """Build prompt for real-world context + empathetic personalization."""
    date_text = "Unknown period"
    if date_range and date_range.get("start") and date_range.get("end"):
        date_text = f"{date_range['start']} to {date_range['end']}"

    lines: list[str] = []
    for bias in top_two_biases:
        bt: BiasType = bias["bias_type"]
        responses = normalized_checkins.get(bt, [])
        response_lines = []
        for item in responses:
            answer = item.get("answer") or "SKIP"
            response_lines.append(f"- {item.get('question')} -> {answer}")
        if not response_lines:
            response_lines.append("- No answers submitted (all skipped or missing).")

        lines.append(
            (
                f"Bias: {bt} ({BIAS_DISPLAY_NAMES[bt]}), score={bias['score']}/100\n"
                f"Responses:\n" + "\n".join(response_lines)
            )
        )

    bias_block = "\n\n".join(lines)

    return f"""You are a financial analyst explaining market context for a trader's behavioral biases.
You are not a financial advisor.

Trader's detected biases:
{bias_block}

Analysis period: {date_text}
Requested max events per bias: {num_events}

Your task for EACH bias:
1) Provide 3-{num_events} real, recent market/economic events from this week or recent past that could emotionally trigger this bias.
2) Provide a plain-language explanation (2-3 sentences) connecting those recent events to why this bias may appear.
3) Provide ONE practical, specific, process-focused action to reduce this bias during similar market conditions.
4) Use the provided YES/NO/SKIP emotional check-in responses to generate a personalized, supportive section.
5) Put the event list in bias_contexts[].market_events because it is shown in the Bias Context section of the UI.

Strict constraints:
- Use only real, recent, publicly documented financial market/economic events.
- Do not invent fictional events, fabricated headlines, or fake dates.
- Focus on emotional and psychological triggers, not just statistics.
- Keep tone empathetic and non-judgmental.
- No financial advice, no buy/sell instructions, no guarantees, no predictions.
- Practical actions must be behavior/process-focused (journaling, pacing, limits, pauses, checklist discipline).
- Personalized explanations must be framed as hypotheses, not facts or diagnoses.
- If exact event dates are uncertain, use approximate dates (for example: "Jan 15") and conservative event wording.

Return JSON only, with this exact structure:
{{
  "bias_contexts": [
    {{
      "bias_type": "OVERTRADING",
      "bias_name": "Overtrading",
      "market_events": [
        {{"date": "Jan 15", "headline": "Market volatility surge with sharp intraday reversals"}},
        {{"date": "Jan 18", "headline": "Major earnings surprises increased sector rotation"}},
        {{"date": "Jan 22", "headline": "Risk-off macro headlines raised uncertainty"}}
      ],
      "connection_explanation": "2-3 sentences connecting these events to emotional triggers for the bias.",
      "practical_takeaway": "One specific process-focused action."
    }}
  ],
  "personalized_sections": [
    {{
      "bias_type": "OVERTRADING",
      "bias_name": "Overtrading",
      "headline": "Possible emotional context",
      "supportive_explanation": "2-3 sentences. Hypothetical, supportive, non-judgmental.",
      "hypothetical_contributors": [
        "Contributor 1",
        "Contributor 2"
      ],
      "gentle_process_habits": [
        "Habit 1",
        "Habit 2",
        "Habit 3"
      ],
      "checkin_summary": {{
        "yes_count": 0,
        "no_count": 0,
        "skipped_count": 3
      }},
      "compassionate_note": "A short reminder that this is context, not judgment."
    }}
  ],
  "global_note": "Educational context only. Not financial advice."
}}"""


def _parse_context_response(response_text: str) -> Optional[dict]:
    """Parse JSON payload from LLM response."""
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start == -1 or end <= 0:
            return None
        data = json.loads(response_text[start:end])
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        logger.error(f"Error parsing context response: {exc}")
        return None


def _fallback_contributors(bias_type: BiasType, responses: list[dict]) -> list[str]:
    label_map: dict[BiasType, list[str]] = {
        "REVENGE_TRADING": [
            "Urgency to recover quickly after losses",
            "Frustration carrying into the next decision",
            "Pressure from feeling behind on the day",
        ],
        "LOSS_AVERSION": [
            "Emotional discomfort when accepting a loss",
            "Hope-driven holding when a position moves against plan",
            "Loss regret outweighing gain satisfaction",
        ],
        "RECENCY_BIAS": [
            "Recent streaks shifting confidence too sharply",
            "Last few outcomes driving size decisions",
            "Pressure to react quickly after fast market moves",
        ],
        "OVERTRADING": [
            "Trading activity used to cope with stress or restlessness",
            "Discomfort when flat and not in a position",
            "Stress increasing trade count beyond plan",
        ],
    }
    selected: list[str] = []
    labels = label_map[bias_type]
    for idx, response in enumerate(responses[:3]):
        if response.get("answer") == "YES":
            selected.append(labels[idx])
    if not selected:
        return labels[:2]
    return selected[:3]


def _make_fallback_bias_context(bias: dict) -> dict:
    bt: BiasType = bias["bias_type"]
    return {
        "bias_type": bt,
        "bias_name": BIAS_DISPLAY_NAMES[bt],
        "score": bias.get("score", 0),
        "market_events": [
            {"date": "Recent period", "headline": "Higher uncertainty and faster intraday shifts"},
            {"date": "Recent period", "headline": "News flow may increase emotional pressure"},
        ],
        "connection_explanation": FALLBACK_CONNECTIONS[bt],
        "practical_takeaway": FALLBACK_HABITS[bt][0],
    }


def _make_fallback_personalized_section(bias: dict, responses: list[dict]) -> dict:
    bt: BiasType = bias["bias_type"]
    summary = _build_checkin_summary(responses)
    contributors = _fallback_contributors(bt, responses)
    return {
        "bias_type": bt,
        "bias_name": BIAS_DISPLAY_NAMES[bt],
        "headline": f"Possible emotional context for {BIAS_DISPLAY_NAMES[bt]}",
        "supportive_explanation": (
            f"{BIAS_DISPLAY_NAMES[bt]} can become more likely when emotional load is high or when daily "
            "life pressure narrows attention. Your check-in answers suggest possible pressure points that are "
            "worth observing without self-judgment."
        ),
        "hypothetical_contributors": contributors,
        "gentle_process_habits": FALLBACK_HABITS[bt],
        "checkin_summary": summary,
        "compassionate_note": (
            "These are hypotheses for reflection, not conclusions about you. "
            "Use them to support process consistency."
        ),
    }


def _get_fallback_context(
    top_two_biases: list[dict],
    normalized_checkins: dict[BiasType, list[dict]],
    date_range: Optional[dict],
) -> dict:
    """Return static fallback context and personalized sections."""
    bias_contexts = [_make_fallback_bias_context(bias) for bias in top_two_biases[:2]]
    personalized_sections = [
        _make_fallback_personalized_section(bias, normalized_checkins.get(bias["bias_type"], []))
        for bias in top_two_biases[:2]
    ]
    return {
        "bias_contexts": bias_contexts,
        "personalized_sections": personalized_sections,
        "global_note": (
            "Educational context only. Emotional check-in output is hypothetical, "
            "non-diagnostic, and not financial advice."
        ),
        "date_range": date_range,
    }


def _merge_with_fallback_defaults(
    parsed: dict,
    top_two_biases: list[dict],
    normalized_checkins: dict[BiasType, list[dict]],
    date_range: Optional[dict],
) -> dict:
    """Keep model output when valid, but guarantee required fields with fallback defaults."""
    fallback = _get_fallback_context(top_two_biases, normalized_checkins, date_range)

    bias_contexts = parsed.get("bias_contexts")
    if not isinstance(bias_contexts, list) or not bias_contexts:
        bias_contexts = fallback["bias_contexts"]

    personalized_sections = parsed.get("personalized_sections")
    if not isinstance(personalized_sections, list) or not personalized_sections:
        personalized_sections = fallback["personalized_sections"]

    # Ensure check-in summaries are always present and accurate
    for section in personalized_sections:
        raw_bt = str(section.get("bias_type", ""))
        bt = normalize_bias_type(raw_bt)
        if bt:
            section["bias_type"] = bt
            section["bias_name"] = BIAS_DISPLAY_NAMES[bt]
            section["checkin_summary"] = _build_checkin_summary(normalized_checkins.get(bt, []))

    return {
        "bias_contexts": bias_contexts[:2],
        "personalized_sections": personalized_sections[:2],
        "global_note": parsed.get("global_note") or fallback["global_note"],
        "date_range": date_range,
    }

