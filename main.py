"""
main.py
-------
CLI entry point for the Resume Screening Agent (Stage 1: core pipeline).
Usage:
    python main.py --jd job_description.txt --resumes ./sample_resumes --out ./output/rankings.csv
Pipeline:
    JD text  -+
              +-> embeddings -> cosine similarity  -+
    Resumes --+                                     +-> final score -> rank -> CSV
              +-> skill extraction -> match ratio ---+
    (optional) --explain flag adds Groq-generated "why this score" text.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from core.parser import load_job_description, load_all_resumes
from core.scorer import rank_resumes
from core.explain import explain_all
def main():
    parser = argparse.ArgumentParser(description="Resume Screening Agent")
    parser.add_argument("--jd", required=True, help="Path to job description (.txt or .pdf)")
    parser.add_argument("--resumes", required=True, help="Directory containing resume PDFs")
    parser.add_argument("--out", default="output/rankings.csv", help="Output CSV path")
    parser.add_argument(
        "--explain", action="store_true",
        help="Generate LLM explanations for each score (requires GROQ_API_KEY in .env)"
    )
    parser.add_argument("--semantic-weight", type=float, default=0.6)
    parser.add_argument("--skill-weight", type=float, default=0.4)
    args = parser.parse_args()
    print(f"Loading job description from {args.jd} ...")
    jd_text = load_job_description(args.jd)
    print(f"Loading resumes from {args.resumes} ...")
    resumes = load_all_resumes(args.resumes)
    if not resumes:
        print("ERROR: No resumes could be loaded. Check the folder and PDF text layers.")
        sys.exit(1)
    print(f"Loaded {len(resumes)} resume(s). Scoring against job description ...")
    ranked = rank_resumes(
        jd_text, resumes,
        semantic_weight=args.semantic_weight,
        skill_weight=args.skill_weight,
    )
    if args.explain:
        print("Generating explanations via Groq (llama-3.3-70b-versatile) ...")
        ranked = explain_all(ranked)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_dict() for r in ranked])
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(out_path, index=False)
    print(f"\nRanking complete. Saved to {out_path}\n")
    print(df[["rank", "filename", "final_score", "semantic_score", "skill_match_ratio"]].to_string(index=False))
if __name__ == "__main__":
    main()
