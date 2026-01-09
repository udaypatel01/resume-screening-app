import os

"""
Central configuration for resume analysis features.
Adjust these values to tune behavior without touching the core logic.
"""

# -------------------------
# Experience configuration
# -------------------------

# Buckets in years of experience
EXPERIENCE_LEVELS = {
    "Fresher": (0, 2),
    "Junior": (2, 5),
    "Mid-Level": (5, 10),
    "Senior": (10, 15),
    "Expert": (15, 99),
}

# -------------------------
# Scoring configuration
# -------------------------

SCORE_WEIGHTS = {
    "skills": 0.4,
    "experience": 0.3,
    "education": 0.2,
    "format": 0.1,
}

# Maximum number of skills considered for a “perfect” skills score
MAX_SKILLS_COUNT = 20

# -------------------------
# Paths / IO configuration
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

SKILLS_DB_PATH = os.path.join(BASE_DIR, "skills_database.json")


def ensure_output_dir() -> str:
    """Ensure the outputs directory exists and return its path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


