"""
Critique Agent
---------------
Evaluates draft emails for quality across multiple dimensions.
Returns scores and feedback. Triggers regeneration if quality is below threshold.
"""

import os
import json
import time
import requests


MAX_API_RETRIES = 5
BASE_WAIT = 10


def get_bedrock_url(model_id: str) -> str:
    """Construct the Bedrock invoke model URL."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    encoded_model = model_id.replace(":", "%3A")
    return f"https://bedrock-runtime.{region}.amazonaws.com/model/{encoded_model}/invoke"


def build_critique_prompt(draft: str, research: dict, examples: list[str]) -> str:
    """Build the critique evaluation prompt."""
    examples_text = "\n\n---\n\n".join(
        [f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples)]
    )

    return f"""You are an expert email quality evaluator. Analyze this outreach email draft.

DRAFT EMAIL:
{draft}

PROSPECT CONTEXT:
- Name: {research['name']}
- Company: {research['company']}
- Hook: {research.get('hook', '')}

FOUNDER'S ACTUAL WRITING EXAMPLES (reference style):
{examples_text}

EVALUATE on these dimensions (score 1-10 for each):

1. voice - How well does the draft match the founder's writing style?
2. personalization - How specific and genuine is the personalization?
3. spam_risk - How likely is this to be flagged as spam? (1=very spammy, 10=very safe)
4. hallucination_risk - How likely does it contain fabricated claims? (1=many fabrications, 10=all verifiable)

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "voice": <score>,
    "personalization": <score>,
    "spam_risk": <score>,
    "hallucination_risk": <score>,
    "overall": <average of all scores>,
    "feedback": "<one sentence of constructive feedback>"
}}

Output ONLY the JSON, nothing else."""


def critique_agent(
    draft: str,
    research: dict,
    examples: list[str],
    model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
) -> dict:
    """Critique a draft email and return quality scores."""
    api_key = os.environ.get("BEDROCK_API_KEY")
    url = get_bedrock_url(model)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0.3,
        "system": "You are a precise email quality evaluator. Always respond with valid JSON only.",
        "messages": [
            {"role": "user", "content": build_critique_prompt(draft, research, examples)}
        ],
    }

    raw_output = None
    for attempt in range(MAX_API_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                raw_output = response.json()["content"][0]["text"].strip()
                break
            elif response.status_code == 429:
                wait_time = BASE_WAIT * (2 ** attempt)
                print(f"  ⚠️  Throttled. Waiting {wait_time}s ({attempt + 1}/{MAX_API_RETRIES})...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Bedrock error {response.status_code}: {response.text}")
                break

        except requests.exceptions.Timeout:
            wait_time = BASE_WAIT * (2 ** attempt)
            print(f"  ⚠️  Timeout. Retrying in {wait_time}s ({attempt + 1}/{MAX_API_RETRIES})...")
            time.sleep(wait_time)

    if raw_output is None:
        return _default_scores("Could not get response from Bedrock.")

    return _parse_scores(raw_output)


def _parse_scores(raw_output: str) -> dict:
    """Parse JSON scores from LLM response, handling markdown fences."""
    # Try direct parse
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    cleaned = raw_output
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract JSON object from anywhere in the text
    start = raw_output.find("{")
    end = raw_output.rfind("}") + 1
    if start != -1 and end > start:
        try:
            scores = json.loads(raw_output[start:end])
            if "overall" not in scores:
                numeric = [v for k, v in scores.items() if k != "feedback" and isinstance(v, (int, float))]
                scores["overall"] = sum(numeric) / len(numeric) if numeric else 5
            return scores
        except json.JSONDecodeError:
            pass

    return _default_scores("Could not parse critique response.")


def _default_scores(feedback: str) -> dict:
    """Return default fallback scores."""
    return {
        "voice": 5,
        "personalization": 5,
        "spam_risk": 5,
        "hallucination_risk": 5,
        "overall": 5,
        "feedback": feedback,
    }


def passes_quality_threshold(scores: dict, threshold: float = 8.0) -> bool:
    """Check if the critique scores pass the quality threshold."""
    return scores.get("overall", 0) >= threshold
