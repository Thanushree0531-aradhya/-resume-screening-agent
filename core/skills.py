"""
skills.py
---------
Simple keyword-based skill extraction and matching.
Why keyword matching alongside embeddings instead of embeddings alone?
Semantic similarity can miss exact must-have requirements (e.g. a JD
that requires "Kubernetes" might match a resume that only mentions
"Docker" as semantically close -- but a recruiter cares about the
literal skill match too). Combining both gives a more defensible score.
This is intentionally simple (substring matching over a curated skill
list) rather than NER/spaCy, so it's transparent and easy to explain
in an interview: "here's exactly why this skill matched."
"""
import re
# A reasonably broad starter list for tech/AI-engineering roles.
# Extend this list or load it from a config file as needed.
DEFAULT_SKILL_LIST = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "react", "vue", "angular", "next.js", "node.js", "fastapi", "django", "flask",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ci/cd",
    "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
    "langchain", "llm", "rag", "embeddings", "prompt engineering",
    "git", "github", "linux", "rest api", "graphql", "microservices",
    "agile", "scrum", "unit testing", "data structures", "algorithms",
]
def extract_skills(text, skill_list=None):
    """
    Find which skills from skill_list appear in the given text.
    Uses word-boundary-aware matching so "go" doesn't match inside
    "google", but "c++" and "c#" (which have no clean word boundary
    on the symbol side) are handled as plain substring checks.
    Args:
        text: resume or JD text to search.
        skill_list: skills to look for; defaults to DEFAULT_SKILL_LIST.
    Returns:
        Set of matched skills (lowercased, as they appear in skill_list).
    """
    if skill_list is None:
        skill_list = DEFAULT_SKILL_LIST
    text_lower = text.lower()
    found = set()
    for skill in skill_list:
        skill_lower = skill.lower()
        if any(ch in skill_lower for ch in ["+", "#", "."]):
            # symbols break \b word-boundary regex; fall back to substring
            if skill_lower in text_lower:
                found.add(skill_lower)
        else:
            pattern = r"\b" + re.escape(skill_lower) + r"\b"
            if re.search(pattern, text_lower):
                found.add(skill_lower)
    return found
def skill_match_score(jd_text, resume_text, skill_list=None):
    """
    Compare required skills (from JD) against skills found in a resume.
    Returns:
        {
            "required_skills": set of skills found in the JD,
            "matched_skills": set of JD skills also found in resume,
            "missing_skills": set of JD skills NOT found in resume,
            "match_ratio": matched / required (0.0 if JD has no skills)
        }
    """
    required = extract_skills(jd_text, skill_list)
    resume_skills = extract_skills(resume_text, skill_list)
    matched = required & resume_skills
    missing = required - resume_skills
    ratio = len(matched) / len(required) if required else 0.0
    return {
        "required_skills": required,
        "matched_skills": matched,
        "missing_skills": missing,
        "match_ratio": ratio,
    }
