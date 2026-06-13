"""
Draft Agent
------------
Generates a personalized outreach email using the founder's voice,
research context, and retrieved email examples.
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


def build_draft_prompt(research: dict, examples: list[str]) -> str:
    """Build the prompt for the LLM to draft an outreach email."""
    examples_text = "\n\n---\n\n".join(
        [f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples)]
    )

    return f"""You are a startup founder writing a cold outreach email.

PROSPECT INFORMATION:
- Name: {research['name']}
- Company: {research['company']}
- Role: {research['role']}
- Industry: {research.get('industry', 'technology')}
- Context: {research.get('summary', '')}
- Personalization Hook: {research.get('hook', '')}

YOUR PAST WRITING EXAMPLES (match this tone and style):
{examples_text}

RULES:
- Under 120 words
- Conversational and genuine tone
- No buzzwords or marketing language
- Exactly one clear call-to-action (a question works best)
- Must reference the personalization hook naturally
- Short paragraphs (1-2 sentences each)
- Sign off as "Founder"

Write the outreach email now. Output ONLY the email text, nothing else."""


def draft_agent(
    research: dict,
    examples: list[str],
    model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
) -> str:
    """Generate a draft outreach email using Amazon Bedrock."""
    api_key = os.environ.get("BEDROCK_API_KEY")
    url = get_bedrock_url(model)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "temperature": 0.7,
        "system": "You are a founder who writes concise, genuine outreach emails. You never sound salesy or use corporate jargon.",
        "messages": [
            {"role": "user", "content": build_draft_prompt(research, examples)}
        ],
    }

    for attempt in range(MAX_API_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                return response.json()["content"][0]["text"].strip()
            elif response.status_code == 429:
                wait_time = BASE_WAIT * (2 ** attempt)
                print(f"  ⚠️  Throttled. Waiting {wait_time}s ({attempt + 1}/{MAX_API_RETRIES})...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Bedrock error {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            wait_time = BASE_WAIT * (2 ** attempt)
            print(f"  ⚠️  Timeout. Retrying in {wait_time}s ({attempt + 1}/{MAX_API_RETRIES})...")
            time.sleep(wait_time)

    raise RuntimeError("Draft agent failed after max retries.")
