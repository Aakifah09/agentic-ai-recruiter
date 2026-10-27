"""Section splitting, date-range parsing and evidence grounding helpers."""

from __future__ import annotations

import re
from datetime import date

SECTION_ALIASES: dict[str, list[str]] = {
    # resume sections
    "summary": ["summary", "profile", "professional summary", "objective", "about me", "career objective"],
    "skills": ["skills", "technical skills", "core skills", "key skills", "competencies", "core competencies", "tools", "technologies", "tech stack", "skills & tools"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "work history", "career history", "internships", "relevant experience"],
    "projects": ["projects", "personal projects", "selected projects", "key projects", "portfolio", "academic projects"],
    "education": ["education", "academic background", "qualifications", "education & training", "academic qualifications"],
    "certifications": ["certifications", "certificates", "licenses", "courses", "training", "certifications & courses"],
    # job description sections
    "responsibilities": ["responsibilities", "key responsibilities", "what you'll do", "what you will do", "the role", "duties", "your role", "day to day", "role responsibilities", "your impact"],
    "required": ["requirements", "required", "required skills", "required qualifications", "must have", "must-have", "essential", "essential skills", "minimum qualifications", "what you bring", "what we're looking for", "what we are looking for", "about you", "qualifications and skills", "you have"],
    "preferred": ["preferred", "preferred qualifications", "preferred skills", "nice to have", "nice-to-have", "bonus", "bonus points", "desirable", "desirable skills", "pluses", "good to have", "it would be great if"],
    "benefits": ["benefits", "what we offer", "perks", "why join us", "compensation"],
    "about": ["about us", "about the company", "who we are", "company overview", "about the team"],
}

_ALIAS_TO_SECTION = {alias: section for section, aliases in SECTION_ALIASES.items() for alias in aliases}


def heading_section(line: str) -> str | None:
    """Return the section name if ``line`` is a heading, else None."""
    if ":" in line:
        prefix = re.sub(r"[^a-z&'\- ]", "", line.split(":", 1)[0].lower()).strip()
        if prefix in _ALIAS_TO_SECTION:
            return _ALIAS_TO_SECTION[prefix]
    stripped = line.strip().strip("#*_=:-").strip().lower()
    stripped = re.sub(r"[^a-z&'\- ]", "", stripped).strip()
    if not stripped or len(stripped.split()) > 6:
        return None
    if stripped in _ALIAS_TO_SECTION:
        return _ALIAS_TO_SECTION[stripped]
    # "Requirements (must have)" / "Preferred qualifications:" style headings
    for alias, section in sorted(_ALIAS_TO_SECTION.items(), key=lambda kv: -len(kv[0])):
        if stripped.startswith(alias) and (line.rstrip().endswith(":") or line.isupper() or line.lstrip().startswith("#")):
            return section
    return None


def split_sections(text: str, default: str = "header") -> dict[str, list[str]]:
    """Split text into {section: [lines]} using heading detection."""
    sections: dict[str, list[str]] = {default: []}
    current = default
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        section = heading_section(line)
        if section:
            current = section
            sections.setdefault(current, [])
            # "Skills: Python, SQL" -> keep the inline content
            if ":" in line and line.split(":", 1)[1].strip():
                sections[current].append(line.split(":", 1)[1].strip())
            continue
        sections.setdefault(current, []).append(line)
    return sections


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MONTH_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_POINT = rf"(?:{_MONTH_RE}\.?\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})"
_END = rf"(?:{_POINT}|present|current|now|today|ongoing)"
DATE_RANGE = re.compile(rf"(?P<start>{_POINT})\s*(?:-|–|—|to|until)\s*(?P<end>{_END})", re.IGNORECASE)


def parse_point(token: str, today: date | None = None) -> date | None:
    today = today or date.today()
    token = token.strip().lower().rstrip(".")
    if token in {"present", "current", "now", "today", "ongoing"}:
        return today
    m = re.fullmatch(rf"({_MONTH_RE})\.?\s+(\d{{4}})", token)
    if m:
        return date(int(m.group(2)), _MONTHS[m.group(1)[:3]], 1)
    m = re.fullmatch(r"(\d{1,2})/(\d{4})", token)
    if m and 1 <= int(m.group(1)) <= 12:
        return date(int(m.group(2)), int(m.group(1)), 1)
    m = re.fullmatch(r"(\d{4})", token)
    if m and 1950 <= int(m.group(1)) <= today.year + 1:
        return date(int(m.group(1)), 1, 1)
    return None


def months_between(start: date, end: date) -> int:
    return max(0, (end.year - start.year) * 12 + end.month - start.month)


def merged_months(intervals: list[tuple[date, date]]) -> int:
    """Total months covered by possibly-overlapping intervals."""
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += months_between(cur_start, cur_end)
            cur_start, cur_end = start, end
    total += months_between(cur_start, cur_end)
    return total


YEARS_RE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:-|to|–)?\s*(?:\d{1,2}\s*)?\+?\s*years?", re.IGNORECASE)


def stated_years(text: str) -> list[float]:
    """Numbers of years mentioned near the word 'experience'."""
    out = []
    for line in re.split(r"[\n.;]", text):
        if re.search(r"experien|track record|background", line, re.IGNORECASE):
            out += [float(m.group(1)) for m in YEARS_RE.finditer(line) if float(m.group(1)) <= 40]
    return out


# --------------------------------------------------------------------------- #
# Evidence grounding
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#]+", text.lower())


def is_grounded(quote: str, source: str, threshold: float = 0.8) -> bool:
    """True if ``quote`` appears (near-)verbatim in ``source``.

    Exact normalised substring match first; otherwise a sliding window of the
    same length must share >= ``threshold`` of the quote's tokens. This rejects
    hallucinated evidence while tolerating whitespace/punctuation drift.
    """
    q = _tokens(quote)
    if not q:
        return False
    src = _tokens(source)
    if " ".join(q) in " ".join(src):
        return True
    n = len(q)
    if n > len(src):
        return False
    need = set(q)
    best = 0.0
    for i in range(0, len(src) - n + 1):
        window = set(src[i : i + n + 2])
        best = max(best, len(need & window) / len(need))
        if best >= threshold:
            return True
    return False


def sentence_around(text: str, start: int, end: int, max_len: int = 220) -> str:
    """The line containing [start, end), trimmed to ``max_len`` characters around the hit."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    line = text[line_start:line_end].strip().lstrip("- ").strip()
    if len(line) <= max_len:
        return line
    rel = start - line_start
    lo = max(0, rel - max_len // 2)
    return ("..." if lo else "") + line[lo : lo + max_len].strip() + "..."
