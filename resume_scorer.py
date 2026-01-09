from typing import Dict, Optional

from config import MAX_SKILLS_COUNT, SCORE_WEIGHTS
from utils import count_bullets


def _clamp_score(score: float) -> float:
    return max(0.0, min(100.0, score))


def compute_skills_score(
    total_resume_skills: int, matched_job_skills: int, has_job_description: bool
) -> float:
    """
    Score based on number of skills and (optionally) match against job description.
    """
    if total_resume_skills <= 0:
        return 0.0

    base = min(total_resume_skills, MAX_SKILLS_COUNT) / MAX_SKILLS_COUNT
    base_score = base * 80.0

    if has_job_description and total_resume_skills > 0:
        match_ratio = matched_job_skills / total_resume_skills
        match_score = match_ratio * 20.0
    else:
        match_score = 0.0

    return _clamp_score(base_score + match_score)


def compute_experience_score(experience_level: str) -> float:
    mapping = {
        "Fresher": 40.0,
        "Junior": 60.0,
        "Mid-Level": 75.0,
        "Senior": 90.0,
        "Expert": 100.0,
    }
    return mapping.get(experience_level, 50.0)


def compute_education_score(text: str) -> float:
    """
    Very simple heuristic based on education keywords.
    """
    lower = text.lower()
    score = 0.0

    if any(k in lower for k in ["phd", "doctorate"]):
        score = 100.0
    elif any(k in lower for k in ["master of", "msc", "m.tech", "mtech", "mba"]):
        score = 90.0
    elif any(
        k in lower
        for k in ["bachelor of", "b.tech", "btech", "b.e", "bsc", "b.sc", "degree"]
    ):
        score = 75.0
    elif "diploma" in lower:
        score = 60.0
    else:
        score = 40.0

    return score


def compute_format_score(text: str) -> float:
    """
    Use bullets/structure to approximate format quality.
    """
    bullets = count_bullets(text)
    if bullets == 0:
        return 50.0
    if bullets >= 40:
        return 95.0
    # Linear scale between 10 bullets (70) and 40 bullets (95)
    if bullets <= 10:
        return 70.0
    ratio = (bullets - 10) / 30.0
    return 70.0 + ratio * 25.0


def score_resume(
    text: str,
    experience_level: str,
    total_resume_skills: int,
    matched_job_skills: int,
    has_job_description: bool,
) -> Dict[str, float]:
    """
    Return detailed component scores plus overall score (0-100).
    """
    skills_score = compute_skills_score(
        total_resume_skills, matched_job_skills, has_job_description
    )
    experience_score = compute_experience_score(experience_level)
    education_score = compute_education_score(text)
    format_score = compute_format_score(text)

    overall = (
        skills_score * SCORE_WEIGHTS["skills"]
        + experience_score * SCORE_WEIGHTS["experience"]
        + education_score * SCORE_WEIGHTS["education"]
        + format_score * SCORE_WEIGHTS["format"]
    )

    return {
        "skills_score": round(skills_score, 2),
        "experience_score": round(experience_score, 2),
        "education_score": round(education_score, 2),
        "format_score": round(format_score, 2),
        "overall_score": round(_clamp_score(overall), 2),
    }


