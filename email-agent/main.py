"""
Main Orchestrator
------------------
Runs the multi-agent outreach pipeline:
  Lead → Research Agent → Retrieval Agent → Draft Agent → Critique Agent → Final Email

Usage:
    python main.py                  # Process all leads
    python main.py --lead 0         # Process a specific lead (by index)
    python main.py --max-retries 3  # Set max regeneration attempts
"""

import os
import csv
import json
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from research_agent import research_agent
from retrieval_agent import FounderVoiceRetriever
from draft_agent import draft_agent
from critique_agent import critique_agent, passes_quality_threshold


# Configuration
LEADS_FILE = os.path.join(os.path.dirname(__file__), "leads.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
QUALITY_THRESHOLD = 8.0
MAX_RETRIES = 3
API_DELAY_SECONDS = 5


def load_leads(filepath: str = LEADS_FILE) -> list[dict]:
    """Load leads from CSV file."""
    leads = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            leads.append(dict(row))
    return leads


def save_email(email: str, lead: dict, scores: dict, output_dir: str = OUTPUT_DIR):
    """Save a finalized email to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    safe_name = lead["name"].lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"TO: {lead['name']} ({lead['company']})\n")
        f.write(f"GENERATED: {datetime.now().isoformat()}\n")
        f.write(f"SCORES: {json.dumps(scores)}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(email)

    return filepath


def process_lead(
    lead: dict,
    retriever: FounderVoiceRetriever,
    max_retries: int = MAX_RETRIES,
    threshold: float = QUALITY_THRESHOLD,
    verbose: bool = True,
) -> dict:
    """Run the full pipeline for a single lead."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Processing: {lead['name']} @ {lead['company']}")
        print(f"{'=' * 60}")

    # Step 1: Research Agent
    if verbose:
        print("\n[1/4] Research Agent...")
    research = research_agent(lead)
    if verbose:
        print(f"  Industry: {research['industry']}")
        print(f"  Hook: {research['hook']}")

    # Step 2: Retrieval Agent (Founder Voice)
    if verbose:
        print("\n[2/4] Retrieval Agent (Founder Voice)...")
    examples = retriever.retrieve(research, top_k=2)
    if verbose:
        print(f"  Retrieved {len(examples)} voice examples.")

    # Step 3 & 4: Draft + Critique loop
    attempt = 0
    best_draft = None
    best_scores = None

    while attempt < max_retries:
        attempt += 1

        # Step 3: Draft Agent
        if verbose:
            print(f"\n[3/4] Draft Agent (attempt {attempt}/{max_retries})...")
        draft = draft_agent(research, examples)
        if verbose:
            print(f"  Draft ({len(draft.split())} words):")
            print(f"  {draft[:100]}...")

        time.sleep(API_DELAY_SECONDS)

        # Step 4: Critique Agent
        if verbose:
            print(f"\n[4/4] Critique Agent...")
        scores = critique_agent(draft, research, examples)
        if verbose:
            print(f"  Scores: {json.dumps(scores, indent=2)}")

        # Track best attempt
        if best_scores is None or scores.get("overall", 0) > best_scores.get("overall", 0):
            best_draft = draft
            best_scores = scores

        # Check quality threshold
        if passes_quality_threshold(scores, threshold):
            if verbose:
                print(f"\n  ✓ PASSED (overall: {scores['overall']:.1f} >= {threshold})")
            break
        else:
            if verbose:
                print(f"\n  ✗ BELOW THRESHOLD (overall: {scores['overall']:.1f} < {threshold})")
                if attempt < max_retries:
                    print(f"  Regenerating...")
            time.sleep(API_DELAY_SECONDS)

    # Save the best email
    saved_path = save_email(best_draft, lead, best_scores)
    if verbose:
        print(f"\n  Saved: {saved_path}")

    return {
        "email": best_draft,
        "scores": best_scores,
        "attempts": attempt,
        "saved_path": saved_path,
        "success": passes_quality_threshold(best_scores, threshold),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Outreach Email Generator")
    parser.add_argument("--lead", type=int, default=None, help="Process specific lead by index")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Max regeneration attempts")
    parser.add_argument("--threshold", type=float, default=QUALITY_THRESHOLD, help="Quality threshold (1-10)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if not os.environ.get("BEDROCK_API_KEY"):
        print("ERROR: BEDROCK_API_KEY is not set in .env file.")
        return

    # Load leads
    leads = load_leads()
    print(f"Loaded {len(leads)} leads from {LEADS_FILE}")

    # Initialize retriever (loads model once)
    print("\nInitializing Founder Voice Retriever...")
    retriever = FounderVoiceRetriever()

    # Process leads
    if args.lead is not None:
        if args.lead >= len(leads):
            print(f"ERROR: Lead index {args.lead} out of range (0-{len(leads)-1})")
            return
        leads_to_process = [leads[args.lead]]
    else:
        leads_to_process = leads

    results = []
    for i, lead in enumerate(leads_to_process):
        result = process_lead(
            lead,
            retriever,
            max_retries=args.max_retries,
            threshold=args.threshold,
            verbose=not args.quiet,
        )
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Processed: {len(results)} leads")
    print(f"Passed quality threshold: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"Output directory: {OUTPUT_DIR}")

    for i, result in enumerate(results):
        status = "✓" if result["success"] else "✗"
        print(
            f"  {status} {leads_to_process[i]['name']} "
            f"(score: {result['scores'].get('overall', 'N/A')}, "
            f"attempts: {result['attempts']})"
        )


if __name__ == "__main__":
    main()
