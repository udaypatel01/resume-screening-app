import io
import zipfile
from typing import Dict, List, Tuple

import pandas as pd
import PyPDF2
import docx

from config import ensure_output_dir
from experience_detector import analyze_experience
from resume_scorer import score_resume
from skill_extractor import extract_skills, match_skills_against_job


SUPPORTED_EXTENSIONS = ("pdf", "docx", "txt")


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(data))
    text_parts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_text_from_docx_bytes(data: bytes) -> str:
    file_like = io.BytesIO(data)
    document = docx.Document(file_like)
    return "\n".join(p.text for p in document.paragraphs)


def _extract_text_from_txt_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    ext = filename.split(".")[-1].lower()
    if ext == "pdf":
        return _extract_text_from_pdf_bytes(data)
    if ext == "docx":
        return _extract_text_from_docx_bytes(data)
    if ext == "txt":
        return _extract_text_from_txt_bytes(data)
    raise ValueError(f"Unsupported file type for batch processing: {ext}")


def process_single_resume(
    filename: str,
    data: bytes,
    svc_model,
    tfidf,
    label_encoder,
    job_description: str | None = None,
) -> Dict[str, object]:
    """
    Process one resume file (already read as bytes) and return a rich result dict.
    """
    text = extract_text_from_bytes(filename, data)

    # Category prediction
    cleaned = " ".join(text.split())
    vectorized = tfidf.transform([cleaned]).toarray()
    predicted = svc_model.predict(vectorized)
    category = label_encoder.inverse_transform(predicted)[0]

    # Skills
    skills_info = extract_skills(text)
    matched_count, total_skills, matched_list = match_skills_against_job(
        skills_info["all"], job_description or ""
    )

    # Experience
    exp_info = analyze_experience(text)

    # Scoring
    scores = score_resume(
        text=text,
        experience_level=str(exp_info.get("level") or "Unknown"),
        total_resume_skills=total_skills,
        matched_job_skills=matched_count,
        has_job_description=bool(job_description),
    )

    return {
        "file_name": filename,
        "category": category,
        "skills_technical": ", ".join(skills_info["technical"]),
        "skills_soft": ", ".join(skills_info["soft"]),
        "skills_all": ", ".join(skills_info["all"]),
        "skills_matched_to_job": ", ".join(matched_list),
        "skills_matched_count": matched_count,
        "skills_total": total_skills,
        "experience_years": exp_info.get("years"),
        "experience_level": exp_info.get("level"),
        "companies": ", ".join(exp_info.get("companies", [])),
        "current_title": exp_info.get("current_title"),
        "score_overall": scores["overall_score"],
        "score_skills": scores["skills_score"],
        "score_experience": scores["experience_score"],
        "score_education": scores["education_score"],
        "score_format": scores["format_score"],
    }


def build_results_dataframe(results: List[Dict[str, object]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def export_results(
    df: pd.DataFrame, base_filename: str = "batch_results"
) -> Tuple[str, str]:
    """
    Export results to Excel and CSV files inside the outputs directory.
    Returns (excel_path, csv_path).
    """
    out_dir = ensure_output_dir()
    excel_path = io.os.path.join(out_dir, f"{base_filename}.xlsx")
    csv_path = io.os.path.join(out_dir, f"{base_filename}.csv")

    # Excel
    df.to_excel(excel_path, index=False)
    # CSV
    df.to_csv(csv_path, index=False)

    return excel_path, csv_path


def iterate_zip_members(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    """
    Return a filtered list of ZipInfo objects for supported resume files.
    """
    members: List[zipfile.ZipInfo] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        ext = name.split(".")[-1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            members.append(info)
    return members


