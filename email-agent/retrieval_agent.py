"""
Retrieval Agent (Founder Voice Retrieval)
------------------------------------------
Uses RAG to retrieve founder email examples most relevant to the
prospect context, preserving the founder's authentic voice.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


EMAILS_DIR = os.path.join(os.path.dirname(__file__), "founder_emails")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_founder_emails(emails_dir: str = EMAILS_DIR) -> list[str]:
    """Load all founder email examples from the emails directory."""
    emails = []
    if not os.path.exists(emails_dir):
        raise FileNotFoundError(f"Emails directory not found: {emails_dir}")

    for filename in sorted(os.listdir(emails_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(emails_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                emails.append(f.read().strip())

    if not emails:
        raise ValueError(f"No .txt email files found in {emails_dir}")

    return emails


class FounderVoiceRetriever:
    """
    Encapsulates the founder voice retrieval pipeline.
    Loads emails and model once, then retrieves examples on demand.
    """

    def __init__(self, emails_dir: str = EMAILS_DIR, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.emails = load_founder_emails(emails_dir)
        print(f"Loaded {len(self.emails)} founder email examples.")
        self.email_embeddings = self.model.encode(self.emails, convert_to_numpy=True)
        print("Email embeddings computed.")

    def retrieve(self, research_context: dict, top_k: int = 2) -> list[str]:
        """Retrieve founder voice examples relevant to the prospect."""
        context_str = (
            f"{research_context.get('company', '')} "
            f"{research_context.get('industry', '')} "
            f"{research_context.get('hook', '')} "
            f"{research_context.get('summary', '')}"
        )

        context_embedding = self.model.encode([context_str], convert_to_numpy=True)
        similarities = cosine_similarity(context_embedding, self.email_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.emails[i] for i in top_indices]
