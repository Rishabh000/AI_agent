# MCRDSE Outreach Agent

A multi-agent system for generating personalized founder outreach emails.

## Architecture

```
Lead Information (leads.csv)
        ↓
  Research Agent      → Structures lead data, infers industry, generates hooks
        ↓
  Retrieval Agent     → RAG-based founder voice retrieval using email examples
        ↓
  Draft Agent         → LLM generates email matching founder's tone
        ↓
  Critique Agent      → Scores on voice/personalization/spam/hallucination
        ↓
  Final Email         → Saved to output/ if score >= 8, else regenerated
```

## Project Structure

```
mcrdse-outreach-agent/
├── founder_emails/        # Founder writing samples for voice matching
│   ├── email1.txt
│   ├── email2.txt
│   └── email3.txt
├── output/                # Generated emails (created at runtime)
├── leads.csv              # Input lead data
├── research_agent.py      # Agent 1: Lead enrichment
├── retrieval_agent.py     # Agent 2: Founder voice RAG
├── draft_agent.py         # Agent 3: Email generation
├── critique_agent.py      # Agent 4: Quality evaluation
├── main.py                # Orchestrator pipeline
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Run the Pipeline

```bash
# Process all leads
python main.py

# Process a specific lead (by index)
python main.py --lead 0

# Custom quality threshold
python main.py --threshold 7.5

# More regeneration attempts
python main.py --max-retries 5

# Quiet mode
python main.py --quiet
```

## Agents

### Research Agent (`research_agent.py`)

Transforms raw lead data into structured context:
- Infers industry from company name and website
- Generates personalization hooks
- Outputs a context dictionary used by downstream agents

### Retrieval Agent (`retrieval_agent.py`)

Uses RAG (Retrieval-Augmented Generation) to maintain founder voice:
- Loads founder email examples from `founder_emails/`
- Creates sentence embeddings using `all-MiniLM-L6-v2`
- Retrieves the most relevant examples via cosine similarity
- No fine-tuning required — voice consistency through examples

### Draft Agent (`draft_agent.py`)

Generates personalized outreach emails using an LLM:
- Combines research context + founder voice examples into a prompt
- Enforces rules: <120 words, conversational, no buzzwords, one CTA
- Uses GPT-4o-mini for cost-effective generation

### Critique Agent (`critique_agent.py`)

Evaluates drafts across four dimensions:
- **Voice consistency** (0-10): Does it sound like the founder?
- **Personalization** (0-10): Is the hook naturally integrated?
- **Spam risk** (0-10, higher = safer): Marketing language check
- **Hallucination risk** (0-10, higher = safer): Fabricated claims check

If overall score < 8, the draft is regenerated (up to max retries).

## Customization

### Add Your Own Voice
Replace files in `founder_emails/` with your actual sent emails.
More examples = better voice matching.

### Add Leads
Edit `leads.csv` with your prospects:
```csv
name,company,role,website
Jane Doe,TechCo,CEO,https://techco.com
```

### Adjust Quality
- `--threshold`: Lower for faster output, higher for better quality
- `--max-retries`: More attempts to hit quality bar

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for draft and critique agents |
