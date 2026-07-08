"""
embeddings.py
-------------
Wraps SentenceTransformers for embedding job descriptions and resumes,
and computing semantic similarity between them.
Model: all-MiniLM-L6-v2
  - 384-dim embeddings, ~80MB, runs fine on CPU.
  - Good tradeoff of speed vs. quality for this use case; no API key
    needed (unlike Gemini embeddings you used in DocMind).
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
@lru_cache(maxsize=1)
def get_model():
    """
    Load and cache the embedding model. Cached so repeated calls
    (e.g. scoring many resumes in a loop) don't reload the model
    from disk each time.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")
def embed_text(text):
    """
    Embed a single piece of text.
    Long resumes are automatically truncated by the tokenizer's max
    sequence length (256 tokens for this model) -- for a first version
    that's fine, since most signal is in the opening summary/skills
    section. If you find scoring is weak, chunk + average later.
    """
    model = get_model()
    return model.encode(text, convert_to_numpy=True)
def embed_many(texts):
    """Embed a batch of texts at once (faster than looping embed_text)."""
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
def semantic_similarity(jd_text, resume_text):
    """
    Compute cosine similarity between a job description and a resume.
    Returns:
        Float in roughly [0, 1] (cosine similarity of embeddings;
        values near 1.0 = very similar, near 0 = unrelated).
    """
    jd_vec = embed_text(jd_text).reshape(1, -1)
    resume_vec = embed_text(resume_text).reshape(1, -1)
    return float(cosine_similarity(jd_vec, resume_vec)[0][0])
def rank_resumes_by_similarity(jd_text, resumes):
    """
    Rank multiple resumes against a single job description.
    Args:
        jd_text: the job description text.
        resumes: dict of filename -> resume text.
    Returns:
        List of (filename, similarity_score) sorted descending by score.
    """
    filenames = list(resumes.keys())
    texts = [resumes[f] for f in filenames]
    jd_vec = embed_text(jd_text).reshape(1, -1)
    resume_vecs = embed_many(texts)
    scores = cosine_similarity(jd_vec, resume_vecs)[0]
    ranked = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)
    return [(name, float(score)) for name, score in ranked]


