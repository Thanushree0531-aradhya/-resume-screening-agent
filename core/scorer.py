"""
scorer.py
---------
Combines semantic similarity (embeddings) and skill matching into one
final score per resume, and ranks all resumes against a job description.
Final score formula (tunable):
    final_score = (semantic_weight * semantic_similarity)
                + (skill_weight * skill_match_ratio)
Default weights: 60% semantic, 40% skill match. Semantic similarity
captures overall fit and phrasing/experience level; skill match ratio
anchors the score to concrete must-have keywords so the model can't be
fooled by resumes that "sound" relevant without the actual skills.
"""
from dataclasses import dataclass, field
from core.embeddings import semantic_similarity
from core.skills import skill_match_score
@dataclass
class ResumeScore:
    filename: str
    semantic_score: float
    skill_match_ratio: float
    matched_skills: set = field(default_factory=set)
    missing_skills: set = field(default_factory=set)
    final_score: float = 0.0
    explanation: str = ""
    def to_dict(self):
        return {
            "filename": self.filename,
            "final_score": round(self.final_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "skill_match_ratio": round(self.skill_match_ratio, 4),
            "matched_skills": ", ".join(sorted(self.matched_skills)),
            "missing_skills": ", ".join(sorted(self.missing_skills)),
            "explanation": self.explanation,
        }
def score_resume(jd_text, resume_text, filename, semantic_weight=0.6, skill_weight=0.4):
    """Score a single resume against a job description."""
    sem_score = semantic_similarity(jd_text, resume_text)
    skill_info = skill_match_score(jd_text, resume_text)
    final = (semantic_weight * sem_score) + (skill_weight * skill_info["match_ratio"])
    return ResumeScore(
        filename=filename,
        semantic_score=sem_score,
        skill_match_ratio=skill_info["match_ratio"],
        matched_skills=skill_info["matched_skills"],
        missing_skills=skill_info["missing_skills"],
        final_score=final,
    )
def rank_resumes(jd_text, resumes, semantic_weight=0.6, skill_weight=0.4):
    """
    Score and rank all resumes against a job description.
    Args:
        jd_text: job description text.
        resumes: dict of filename -> resume text.
        semantic_weight / skill_weight: must sum to 1.0 (not enforced,
            but scores won't be interpretable as 0-1 if they don't).
    Returns:
        List of ResumeScore, sorted descending by final_score.
    """
    scores = [
        score_resume(jd_text, text, fname, semantic_weight, skill_weight)
        for fname, text in resumes.items()
    ]
    return sorted(scores, key=lambda s: s.final_score, reverse=True)
