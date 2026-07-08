---
title: Resume Screening Agent
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
license: mit
---
# Resume Screening Agent
An AI-powered resume screening tool that ranks candidate resumes against a job description using a combination of semantic similarity (sentence embeddings) and keyword-based skill matching. Includes an optional LLM-generated explanation for each score.
**Live demo:**
- Frontend: https://resume-screening-desk.netlify.app
- Backend API: https://thanushreet-resume-screening-agent.hf.space
- API docs (Swagger UI): https://thanushreet-resume-screening-agent.hf.space/docs
> Note: the backend runs on a free Hugging Face Space, which sleeps after inactivity. The first request after idle time may take 30-60 seconds while the container wakes up.
---
## How It Works
Each resume is scored against the job description using two independent signals, then combined into a final weighted score:
final_score = (semantic_weight * semantic_similarity) + (skill_weight * skill_match_ratio)
**1. Semantic similarity** - measures how closely the overall meaning of a resume matches the job description, using all-MiniLM-L6-v2 sentence embeddings and cosine similarity. Since the model has a 256-token limit, long resumes are split into overlapping chunks, each embedded separately; the single strongest chunk-to-chunk match against the job description is used, so one relevant section (e.g. the skills or experience section) can carry the score even if other sections (education, contact info) are unrelated.
**2. Skill match ratio** - extracts required skills from the job description and checks which of them literally appear in the resume, using word-boundary-aware substring matching against a curated skill list. This anchors the score to concrete, defensible keyword matches that a recruiter would recognize, independent of phrasing or tone.
**3. Explanation (optional)** - when enabled, an LLM (Llama 3.3 70B via Groq) generates a short natural-language explanation of why a resume received its score, referencing the matched/missing skills and the semantic fit.
Default weighting is 60% semantic similarity, 40% skill match - tunable via the API.
---
## Tech Stack
| Layer | Technology |
|---|---|
| Frontend | React + Vite |
| Backend API | FastAPI (Python) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Skill matching | Regex-based keyword extraction |
| Explanations | Groq API (Llama 3.3 70B) |
| PDF parsing | pdfplumber |
| Frontend hosting | Netlify |
| Backend hosting | Hugging Face Spaces (Docker) |
---
## Project Structure
resume-screening-agent/
|-- api.py                  (FastAPI app, deployed to Hugging Face)
|-- main.py                 (CLI entry point for local batch scoring)
|-- requirements.txt
|-- Dockerfile               (Container spec for Hugging Face Spaces)
|-- core/
|   |-- parser.py            (PDF text extraction)
|   |-- embeddings.py        (Chunked sentence embeddings + cosine similarity)
|   |-- skills.py            (Keyword-based skill extraction and matching)
|   |-- scorer.py            (Combines scores, ranks resumes)
|   |-- explain.py           (LLM-generated score explanations via Groq)
|-- resume-frontend/
    |-- src/
    |   |-- App.jsx           (Main UI: job description input, resume upload, results)
    |   |-- App.css
    |-- package.json
---
## Running Locally
### Backend
cd resume-screening-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt --break-system-packages
Create a .env file with your Groq API key (only needed for the --explain feature):
GROQ_API_KEY=your_key_here
Run the API:
uvicorn api:app --reload
The API will be live at http://127.0.0.1:8000, with interactive docs at http://127.0.0.1:8000/docs.
Or use the CLI directly (no server needed):
python main.py --jd job_description.txt --resumes ./sample_resumes --out ./output/rankings.csv --explain
### Frontend
cd resume-frontend
npm install
npm run dev
Update API_URL in src/App.jsx to point at your local backend (http://127.0.0.1:8000) or the deployed one.
---
## API Reference
### POST /score
Scores and ranks uploaded resumes against a job description.
Form fields:
| Field | Type | Required | Description |
|---|---|---|---|
| job_description | string | yes | Full job description text |
| resumes | file[] | yes | One or more resume PDFs |
| explain | boolean | no | Generate LLM explanations (default: false) |
| semantic_weight | float | no | Weight for semantic similarity (default: 0.6) |
| skill_weight | float | no | Weight for skill match (default: 0.4) |
Response: array of ranked results, each containing rank, filename, final_score, semantic_score, skill_match_ratio, matched_skills, missing_skills, explanation, and skipped/skip_reason for resumes that couldn't be processed (e.g. scanned PDFs with no extractable text).
---
## Deployment
- Backend is containerized via Dockerfile and deployed to Hugging Face Spaces (Docker SDK). Push to the Space's main branch to trigger a rebuild.
- Frontend is built with npm run build and deployed as a static site via the Netlify CLI (netlify deploy --prod --dir=dist).
---
## Known Limitations
- Skill matching uses a fixed, curated keyword list (core/skills.py) - it won't catch synonyms or skills phrased differently than the list expects.
- Scanned/image-only PDFs (no text layer) can't be parsed and are skipped, since OCR is out of scope.
- The embedding model is general-purpose and CPU-sized; semantic similarity is a useful directional signal but not a precise, deterministic measure.
- Free-tier Hugging Face Spaces sleep after inactivity, adding a cold-start delay to the first request.
---
## License
MIT
