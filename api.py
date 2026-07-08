"""
api.py
------
FastAPI wrapper around the existing CLI pipeline (core.parser, core.scorer,
core.explain). Exposes POST /score for the React frontend.
Run with:
    uvicorn api:app --reload
"""
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from core.parser import extract_text_from_pdf, EmptyPDFTextError
from core.scorer import rank_resumes
from core.explain import explain_all
app = FastAPI(title="Resume Screening Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/score")
async def score(
    job_description: str = Form(...),
    explain: str = Form("false"),
    resumes: List[UploadFile] = File(...),
):
    explain_flag = explain.lower() == "true"
    resume_texts = {}
    skipped = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for upload in resumes:
            dest = tmp_path / upload.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            try:
                resume_texts[upload.filename] = extract_text_from_pdf(dest)
            except EmptyPDFTextError:
                skipped.append({
                    "filename": upload.filename,
                    "skipped": True,
                    "skip_reason": "No extractable text (likely a scanned PDF).",
                })
            except Exception as e:
                skipped.append({
                    "filename": upload.filename,
                    "skipped": True,
                    "skip_reason": f"Failed to read PDF: {e}",
                })
        if not resume_texts:
            return skipped
        ranked = rank_resumes(job_description, resume_texts)
        if explain_flag:
            ranked = explain_all(ranked)
    results = []
    for i, r in enumerate(ranked, start=1):
        d = r.to_dict()
        d["rank"] = i
        d["skipped"] = False
        d["matched_skills"] = (
            [s.strip() for s in d["matched_skills"].split(",") if s.strip()]
        )
        d["missing_skills"] = (
            [s.strip() for s in d["missing_skills"].split(",") if s.strip()]
        )
        results.append(d)
    return results + skipped
