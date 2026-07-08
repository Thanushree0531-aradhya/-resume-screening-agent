"""
app.py
------
FastAPI backend for the Resume Screening Agent.
Wraps the same core/ pipeline used by main.py (CLI) behind an HTTP API,
so a frontend (React, or anything else) can POST a job description and
resume PDFs and get back ranked JSON results.
Run locally:
    uvicorn app:app --reload --port 8000
Endpoints:
    GET  /health              -> simple liveness check
    POST /score                -> upload JD text + resume PDFs, get ranked results
CORS is wide open here for local development. Before deploying (e.g. to
Hugging Face Spaces, like your DocMind backend), tighten allow_origins
to your actual frontend domain.
"""
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.parser import extract_text_from_pdf, EmptyPDFTextError
from core.scorer import rank_resumes
from core.explain import explain_all
app = FastAPI(title="Resume Screening Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class ScoreResult(BaseModel):
    rank: int
    filename: str
    final_score: float
    semantic_score: float
    skill_match_ratio: float
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str = ""
    skipped: bool = False
    skip_reason: str = ""
@app.get("/health")
def health():
    return {"status": "ok"}
@app.post("/score", response_model=list[ScoreResult])
async def score_resumes(
    job_description: str = Form(..., description="Job description text"),
    resumes: list[UploadFile] = File(..., description="Resume PDF files"),
    explain: bool = Form(False, description="Generate LLM explanations via Groq"),
    semantic_weight: float = Form(0.6),
    skill_weight: float = Form(0.4),
):
    """
    Score and rank uploaded resumes against a job description.
    """
    if not job_description.strip():
        raise HTTPException(400, "job_description cannot be empty")
    if not resumes:
        raise HTTPException(400, "At least one resume PDF is required")
    resume_texts = {}
    skipped = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for upload in resumes:
            if not upload.filename.lower().endswith(".pdf"):
                skipped.append({"filename": upload.filename, "reason": "Not a PDF file"})
                continue
            dest = tmp_path / upload.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            try:
                resume_texts[upload.filename] = extract_text_from_pdf(dest)
            except EmptyPDFTextError as e:
                skipped.append({"filename": upload.filename, "reason": str(e)})
            except Exception as e:
                skipped.append({"filename": upload.filename, "reason": f"Failed to read: {e}"})
    if not resume_texts:
        raise HTTPException(422, f"No resumes could be processed. Skipped: {skipped}")
    ranked = rank_resumes(
        job_description, resume_texts,
        semantic_weight=semantic_weight,
        skill_weight=skill_weight,
    )
    if explain:
        ranked = explain_all(ranked)
    results = []
    for i, r in enumerate(ranked, start=1):
        d = r.to_dict()
        results.append(ScoreResult(
            rank=i,
            filename=d["filename"],
            final_score=d["final_score"],
            semantic_score=d["semantic_score"],
            skill_match_ratio=d["skill_match_ratio"],
            matched_skills=[s for s in d["matched_skills"].split(", ") if s],
            missing_skills=[s for s in d["missing_skills"].split(", ") if s],
            explanation=d["explanation"],
        ))
    for s in skipped:
        results.append(ScoreResult(
            rank=0,
            filename=s["filename"],
            final_score=0.0,
            semantic_score=0.0,
            skill_match_ratio=0.0,
            matched_skills=[],
            missing_skills=[],
            skipped=True,
            skip_reason=s["reason"],
        ))
    return results
