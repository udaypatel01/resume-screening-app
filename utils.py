import re
from typing import Tuple


def clean_text_basic(text: str) -> str:
    """Lightweight text cleaning for analysis (keep it readable)."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_bullets(text: str) -> int:
    """Approximate bullet/structure count for format quality scoring."""
    if not text:
        return 0
    bullet_patterns = [
        r"^\s*[-*]\s+",
        r"^\s*\d+\.\s+",
        r"•\s+",
    ]
    count = 0
    for line in text.splitlines():
        for pattern in bullet_patterns:
            if re.search(pattern, line):
                count += 1
                break
    return count


def estimate_reading_length(text: str) -> Tuple[int, int]:
    """Return (word_count, char_count)."""
    if not text:
        return 0, 0
    words = text.split()
    return len(words), len(text)


def extract_emails(text: str) -> list:
    """Extract email addresses from text."""
    if not text:
        return []
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.findall(pattern, text)


