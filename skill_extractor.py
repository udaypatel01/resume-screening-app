import json
import re
from typing import Dict, List, Tuple, Set

from config import SKILLS_DB_PATH

try:
    import spacy  # type: ignore

    _SPACY_AVAILABLE = True
except Exception:
    spacy = None
    _SPACY_AVAILABLE = False

_NLP = None


def _load_spacy_model():
    """Lazy-load a small English model if available, otherwise return None."""
    global _NLP
    if not _SPACY_AVAILABLE:
        return None
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = None
    return _NLP


def _load_skills_database() -> Dict[str, List[str]]:
    with open(SKILLS_DB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize skills to lowercase for matching
    return {
        "technical": [s.lower() for s in data.get("technical", [])],
        "soft": [s.lower() for s in data.get("soft", [])],
    }


_SKILLS_DB = _load_skills_database()


def _normalize_text(text: str) -> str:
    return text.lower()


def extract_technical_skills(text: str) -> List[str]:
    text_norm = _normalize_text(text)
    found: Set[str] = set()
    for skill in _SKILLS_DB["technical"]:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_norm):
            found.add(skill)
    return sorted(found)


def extract_soft_skills(text: str) -> List[str]:
    text_norm = _normalize_text(text)
    found: Set[str] = set()
    for skill in _SKILLS_DB["soft"]:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_norm):
            found.add(skill)
    return sorted(found)


def extract_skills(text: str) -> Dict[str, List[str]]:
    """
    High-level helper to extract both technical and soft skills.
    Returns dict with keys: technical, soft, all.
    """
    technical = extract_technical_skills(text)
    soft = extract_soft_skills(text)
    all_skills = sorted(set(technical + soft))
    return {"technical": technical, "soft": soft, "all": all_skills}


def match_skills_against_job(
    resume_skills: List[str], job_description: str
) -> Tuple[int, int, List[str]]:
    """
    Compare resume skills against job description text.
    Returns (matched_count, total_unique_resume_skills, matched_skills_list).
    """
    if not job_description:
        return 0, len(set(resume_skills)), []

    jd_norm = job_description.lower()
    matched: Set[str] = set()
    for skill in resume_skills:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, jd_norm):
            matched.add(skill)

    return len(matched), len(set(resume_skills)), sorted(matched)


