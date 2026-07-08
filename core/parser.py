"""
parser.py
---------
Extracts raw text from resume PDFs and job description files.
Uses pdfplumber (pure Python, no external binaries needed -- unlike
Tesseract/Poppler which caused you path headaches in DocMind on Windows).
Note: this does NOT do OCR. If a resume PDF is a scanned image with no
text layer, extract_text_from_pdf() will return an empty string. We
detect that case explicitly so the caller can skip/flag the file
instead of crashing downstream (this is exactly the bug you hit in
DocMind -- empty text reaching the embedding step).
"""
from pathlib import Path
import pdfplumber
class EmptyPDFTextError(Exception):
    """Raised when a PDF has no extractable text layer (likely scanned)."""
    pass
def extract_text_from_pdf(pdf_path):
    """
    Extract all text from a PDF file.
    Args:
        pdf_path: path to the PDF file.
    Returns:
        Full extracted text, page breaks joined with newlines.
    Raises:
        FileNotFoundError: if the path doesn't exist.
        EmptyPDFTextError: if no text could be extracted (likely a
            scanned/image-only PDF -- would need OCR, which is out of
            scope for this module by design).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise EmptyPDFTextError(
            f"No extractable text in {pdf_path.name}. "
            "This is likely a scanned/image PDF and needs OCR "
            "(out of scope here -- flag and skip it instead of scoring)."
        )
    return full_text
def load_job_description(jd_path):
    """
    Load a job description from a .txt or .pdf file.
    Args:
        jd_path: path to a .txt or .pdf job description file.
    Returns:
        The job description text.
    """
    jd_path = Path(jd_path)
    if not jd_path.exists():
        raise FileNotFoundError(f"Job description not found: {jd_path}")
    if jd_path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(jd_path)
    return jd_path.read_text(encoding="utf-8").strip()
def load_all_resumes(resume_dir):
    """
    Extract text from every PDF in a directory.
    Args:
        resume_dir: directory containing resume PDFs.
    Returns:
        Dict mapping filename -> extracted text. Files that fail to
        extract (e.g. scanned PDFs) are skipped and reported via
        print(), not silently dropped.
    """
    resume_dir = Path(resume_dir)
    resumes = {}
    pdf_files = sorted(resume_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"WARNING: No PDF files found in {resume_dir}")
        return resumes
    for pdf_file in pdf_files:
        try:
            resumes[pdf_file.name] = extract_text_from_pdf(pdf_file)
        except EmptyPDFTextError as e:
            print(f"WARNING: Skipping {pdf_file.name}: {e}")
        except Exception as e:
            print(f"WARNING: Failed to read {pdf_file.name}: {e}")
    return resumes
