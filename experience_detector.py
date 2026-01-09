import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import EXPERIENCE_LEVELS


def _extract_year_spans(text: str) -> List[Tuple[float, float]]:
    """Extract explicit year ranges like '2-4 years', '3 – 5 years' etc."""
    spans: List[Tuple[float, float]] = []
    pattern = r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s+years?"
    for match in re.finditer(pattern, text.lower()):
        start = float(match.group(1))
        end = float(match.group(2))
        spans.append((start, end))
    return spans


def _extract_single_years(text: str) -> List[float]:
    """Extract mentions like '3 years', '5+ years'."""
    years: List[float] = []
    pattern = r"(\d+(?:\.\d+)?)\s*\+?\s+years?"
    for match in re.finditer(pattern, text.lower()):
        years.append(float(match.group(1)))
    return years


def _extract_date_ranges(text: str) -> List[Tuple[float, float]]:
    """Extract experience from date ranges like 'Jan 2020 - Dec 2023'."""
    
    # Pattern for date ranges: "MMM YYYY - MMM YYYY" or "MM/YYYY - MM/YYYY"
    date_patterns = [
        r"([A-Za-z]{3,9})\s+(\d{4})\s*[-–]\s*([A-Za-z]{3,9})?\s*(\d{4})",  # Jan 2020 - Dec 2023
        r"(\d{1,2})[/-](\d{4})\s*[-–]\s*(\d{1,2})?[/-]?(\d{4})",  # 01/2020 - 12/2023
    ]
    
    ranges: List[Tuple[float, float]] = []
    current_year = datetime.now().year
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                if len(match.groups()) == 4:
                    if match.group(1).isdigit():  # MM/YYYY format
                        start_month, start_year = int(match.group(1)), int(match.group(2))
                        end_month = int(match.group(3)) if match.group(3) else 12
                        end_year = int(match.group(4)) if match.group(4) else current_year
                    else:  # MMM YYYY format
                        start_month_str, start_year_str = match.group(1), match.group(2)
                        end_month_str = match.group(3) if match.group(3) else "Dec"
                        end_year_str = match.group(4) if match.group(4) else str(current_year)
                        
                        # Convert month names to numbers
                        month_map = {
                            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                        }
                        start_month = month_map.get(start_month_str[:3].lower(), 1)
                        start_year = int(start_year_str)
                        end_month = month_map.get(end_month_str[:3].lower(), 12)
                        end_year = int(end_year_str)
                    
                    # Calculate years
                    start_date = start_year + (start_month - 1) / 12.0
                    end_date = end_year + (end_month - 1) / 12.0
                    years_diff = end_date - start_date
                    if years_diff > 0:
                        ranges.append((start_date, end_date))
            except (ValueError, AttributeError):
                continue
    
    return ranges


def estimate_years_of_experience(text: str) -> Optional[float]:
    """
    Estimate total years of experience from resume text.
    Enhanced: uses date ranges, year spans, and single year mentions.
    """
    spans = _extract_year_spans(text)
    singles = _extract_single_years(text)
    date_ranges = _extract_date_ranges(text)

    candidates: List[float] = []
    
    # From explicit year spans
    candidates.extend([max(start, end) for start, end in spans])
    
    # From single year mentions
    candidates.extend(singles)
    
    # From date ranges - calculate total years
    if date_ranges:
        # Calculate total years from all date ranges
        total_years = 0.0
        for start, end in date_ranges:
            total_years += (end - start)
        candidates.append(total_years)
    
    # Also try to find "X years of experience" patterns
    exp_pattern = r"(\d+(?:\.\d+)?)\s+years?\s+of\s+experience"
    for match in re.finditer(exp_pattern, text.lower()):
        candidates.append(float(match.group(1)))

    if not candidates:
        return None

    # Return the maximum found, but also consider sum if multiple ranges
    return max(candidates)


def classify_experience_level(years: Optional[float]) -> str:
    if years is None:
        return "Unknown"
    for level, (low, high) in EXPERIENCE_LEVELS.items():
        if low <= years < high:
            return level
    return "Unknown"


def extract_company_names(text: str, max_companies: int = 10) -> List[str]:
    """
    Enhanced company name extraction: looks for multiple patterns including
    'at X', 'with X', 'Company: X', employment sections, etc.
    """
    text_lines = text.splitlines()
    companies: List[str] = []
    seen = set()
    
    # Enhanced patterns
    patterns = [
        r"(?:at|with|@)\s+([A-Z][A-Za-z0-9&.,\-' ]{2,50})",  # at Company Name
        r"Company[:\-]\s*([A-Z][A-Za-z0-9&.,\-' ]{2,50})",  # Company: Name
        r"Employer[:\-]\s*([A-Z][A-Za-z0-9&.,\-' ]{2,50})",  # Employer: Name
        r"Organization[:\-]\s*([A-Z][A-Za-z0-9&.,\-' ]{2,50})",  # Organization: Name
        r"([A-Z][A-Za-z0-9&.,\-' ]{2,50})\s+(?:Inc|LLC|Ltd|Corp|Corporation|Technologies|Tech|Solutions|Systems)",  # Company Inc
        r"(?:Worked at|Employed at|Experience at)\s+([A-Z][A-Za-z0-9&.,\-' ]{2,50})",  # Worked at Company
    ]
    
    # Common words to exclude
    exclude_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'cannot'
    }

    for line in text_lines:
        line_clean = line.strip()
        if len(line_clean) < 3:
            continue
            
        for pat in patterns:
            for match in re.finditer(pat, line, re.IGNORECASE):
                name = match.group(1).strip()
                # Clean up the name
                name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
                name = name.rstrip('.,;:')  # Remove trailing punctuation
                
                # Filter out very short or common words
                if len(name) < 2:
                    continue
                words = name.split()
                if len(words) == 1 and words[0].lower() in exclude_words:
                    continue
                
                # Check if it looks like a company name (starts with capital, reasonable length)
                if name and name not in seen and len(name) <= 50:
                    # Additional validation: should have at least one letter
                    if re.search(r'[A-Za-z]', name):
                        companies.append(name)
                        seen.add(name)
                        if len(companies) >= max_companies:
                            return companies[:max_companies]
    
    return companies[:max_companies]


def extract_current_title(text: str) -> Optional[str]:
    """
    Enhanced current/recent job title extraction.
    Looks for title patterns in the top section and experience section.
    """
    keywords = [
        "engineer", "developer", "manager", "analyst", "consultant",
        "specialist", "lead", "architect", "intern", "director",
        "executive", "coordinator", "assistant", "associate", "senior",
        "junior", "principal", "chief", "head", "vice", "president",
        "officer", "designer", "programmer", "scientist", "researcher",
        "administrator", "supervisor", "technician", "advisor"
    ]

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    
    # Check first 50 lines (header/experience section)
    top_lines = lines[:50]
    
    # Patterns for job titles
    title_patterns = [
        r"^(?:Senior|Junior|Lead|Principal|Chief|Head|Associate|Assistant)?\s*"
        r"([A-Z][A-Za-z\s]+(?:Engineer|Developer|Manager|Analyst|Consultant|Specialist|Architect|Designer|Programmer|Scientist|Researcher|Director|Officer|Coordinator|Administrator|Supervisor|Technician|Advisor|Intern))",
        r"Title[:\-]\s*([A-Z][A-Za-z\s]{5,50})",
        r"Position[:\-]\s*([A-Z][A-Za-z\s]{5,50})",
        r"Role[:\-]\s*([A-Z][A-Za-z\s]{5,50})",
    ]
    
    # First, try pattern matching
    for line in top_lines:
        for pattern in title_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) >= 5 and len(title) <= 80:
                    return title
    
    # Fallback: keyword-based search
    for line in top_lines:
        lower = line.lower()
        # Check if line contains title keywords and is reasonably short (likely a title)
        if any(k in lower for k in keywords) and len(line) <= 80:
            # Clean up the line
            title = line.strip()
            # Remove common prefixes/suffixes that aren't part of title
            title = re.sub(r'^(?:at|with|@)\s+', '', title, flags=re.IGNORECASE)
            title = re.sub(r'\s+(?:Inc|LLC|Ltd|Corp)\.?$', '', title, flags=re.IGNORECASE)
            if len(title) >= 5:
                return title
    
    return None


def analyze_experience(text: str) -> Dict[str, object]:
    """
    High-level helper aggregating experience info.
    Returns dict with: years, level, companies, current_title.
    """
    years = estimate_years_of_experience(text)
    level = classify_experience_level(years)
    companies = extract_company_names(text)
    current_title = extract_current_title(text)

    return {
        "years": years,
        "level": level,
        "companies": companies,
        "current_title": current_title,
    }


