"""
Research Agent
--------------
Turns raw lead data into structured context for email personalization.
"""


def research_agent(lead: dict) -> dict:
    """
    Takes a lead dictionary and returns enriched research context.

    In production, this would call external APIs (LinkedIn, Crunchbase, etc.)
    For now, it generates structured context from available lead data.
    """
    domain = lead.get("website", "")
    industry = _infer_industry(lead["company"], domain)
    hook = _generate_hook(lead["company"], industry)

    return {
        "name": lead["name"],
        "company": lead["company"],
        "role": lead["role"],
        "website": lead.get("website", ""),
        "industry": industry,
        "summary": f"{lead['company']} operates in {industry}.",
        "hook": hook,
    }


def _infer_industry(company: str, website: str) -> str:
    """Infer industry from company name and website domain."""
    keywords = {
        "health": "digital health",
        "care": "healthcare technology",
        "med": "medical technology",
        "ai": "artificial intelligence",
        "outcome": "health outcomes analytics",
        "stack": "health infrastructure",
        "loop": "care coordination",
        "mind": "mental wellness",
    }
    combined = (company + " " + website).lower()
    for keyword, industry in keywords.items():
        if keyword in combined:
            return industry
    return "technology"


def _generate_hook(company: str, industry: str) -> str:
    """Generate a personalization hook based on company and industry."""
    hooks = {
        "digital health": "Recently expanding patient engagement initiatives.",
        "healthcare technology": "Building next-gen tools for care delivery.",
        "medical technology": "Innovating at the intersection of clinical workflows and technology.",
        "artificial intelligence": "Applying AI to improve real-world health outcomes.",
        "health outcomes analytics": "Pioneering outcomes-based approaches in healthcare.",
        "health infrastructure": "Rethinking the infrastructure layer for modern healthcare.",
        "care coordination": "Tackling care coordination across fragmented systems.",
        "mental wellness": "Addressing the growing demand for accessible mental health tools.",
    }
    return hooks.get(industry, "Doing interesting work in their space.")
