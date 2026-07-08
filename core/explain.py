"""
explain.py
----------
Uses Groq's llama-3.3-70b-versatile to generate a short, grounded
explanation for why a candidate received their score.
Anti-hallucination design: the LLM is NEVER asked to invent a score or
judge fit from scratch. It's only given the *already computed* numbers
(semantic score, matched/missing skills) and asked to explain them in
plain English. This mirrors the "bonus" idea in your project plan --
GPT explains, it doesn't decide.
IMPORTANT: reads GROQ_API_KEY from environment / .env file. Never
hardcode the key in this file or paste it into chat -- load it via
python-dotenv from a local .env that's in your .gitignore.
"""
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
_client = None
def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not found. Create a .env file with "
                "GROQ_API_KEY=your_key_here (and make sure .env is in .gitignore)."
            )
        _client = Groq(api_key=api_key)
    return _client
def explain_score(filename, semantic_score, skill_match_ratio, matched_skills, missing_skills):
    """
    Generate a 2-3 sentence explanation of a resume's score.
    Grounded strictly in the numbers/skills passed in -- the prompt
    explicitly forbids inferring anything not given, to avoid the LLM
    fabricating candidate details it was never shown the full resume for.
    """
    client = get_client()
    matched_str = ", ".join(sorted(matched_skills)) or "none"
    missing_str = ", ".join(sorted(missing_skills)) or "none"
    prompt = f"""You are explaining an automated resume-screening score to a recruiter.
Use ONLY the data below. Do not invent details about the candidate you
were not given. Be concise: 2-3 sentences, plain English, no headers.
Candidate file: {filename}
Semantic similarity to job description: {semantic_score:.2f} (0 to 1 scale)
Skill match ratio: {skill_match_ratio:.2f} (0 to 1 scale)
Matched required skills: {matched_str}
Missing required skills: {missing_str}
Explain why this candidate received this score, referencing the matched
and missing skills specifically."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()
def explain_all(scores):
    """
    Add an explanation to each ResumeScore in place.
    Args:
        scores: list of core.scorer.ResumeScore objects (mutated in place).
    Returns:
        The same list, with .explanation filled in on each item.
    """
    for score in scores:
        try:
            score.explanation = explain_score(
                filename=score.filename,
                semantic_score=score.semantic_score,
                skill_match_ratio=score.skill_match_ratio,
                matched_skills=score.matched_skills,
                missing_skills=score.missing_skills,
            )
        except Exception as e:
            score.explanation = f"(explanation unavailable: {e})"
    return scores
