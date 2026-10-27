"""Privacy and fairness utilities.

* ``redact`` strips contact details and protected-characteristic statements from
  resume text before any agent (or LLM) sees it.
* ``PROTECTED_TERMS`` is used by the compliance agent to scan generated output
  and job descriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Digit runs joined by single separators; only counted as a phone number with 9+ digits,
# so year ranges such as 2019-2021 survive.
_PHONE = re.compile(r"(?<![\w-])\+?\(?\d[\d()]*(?:[\s.-]\(?\d[\d()]*)+(?![\w-])|(?<![\w-])\+?\d{9,15}(?![\w-])")
_URL = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S+", re.IGNORECASE)

# Lines that state a protected characteristic. Whole line is removed.
_PROTECTED_LINE = re.compile(
    r"^.*\b(date of birth|d\.o\.b|dob|age|gender|sex|marital status|married|single|divorced|nationality|"
    r"citizenship|religion|ethnicity|race|visa status|place of birth|number of children|dependants|"
    r"pronouns|disability|blood group|passport)\b\s*[:\-].*$",
    re.IGNORECASE | re.MULTILINE,
)
_ADDRESS = re.compile(r"^\s*(address|location|lives in)\s*[:\-].*$", re.IGNORECASE | re.MULTILINE)
_HEADING_WORDS = {
    "resume", "curriculum", "vitae", "cv", "profile", "summary", "experience", "skills", "education",
    "objective", "contact", "projects", "certifications",
}


@dataclass
class RedactionResult:
    text: str
    name: str | None
    redactions: list[str] = field(default_factory=list)


def guess_name(text: str) -> str | None:
    """Guess the candidate's name from a 'Name:' line or the first line of the resume."""
    m = re.search(r"^\s*(?:full\s+)?name\s*[:\-]\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    for line in text.splitlines()[:5]:
        line = line.strip().strip("#").strip()
        if not line:
            continue
        # Drop trailing "| Data Scientist" style headline.
        candidate = re.split(r"\s[|•\-–—]\s|,", line)[0].strip()
        words = candidate.split()
        if (
            2 <= len(words) <= 4
            and all(re.fullmatch(r"[A-Z][a-zA-Z'\-.]+", w) for w in words)
            and not ({w.lower().strip(".") for w in words} & _HEADING_WORDS)
        ):
            return candidate
        break
    return None


def redact(text: str, alias: str = "[CANDIDATE]") -> RedactionResult:
    """Remove contact data, protected-characteristic lines and the candidate's name."""
    redactions: list[str] = []
    name = guess_name(text)

    def _sub(pattern: re.Pattern[str], label: str, replacement: str, value: str) -> str:
        new, count = pattern.subn(replacement, value)
        if count:
            redactions.append(label)
        return new

    out = _sub(_EMAIL, "email", "[EMAIL]", text)
    out = _sub(_URL, "url", "[URL]", out)
    out = _sub(_PROTECTED_LINE, "protected characteristic", "", out)
    out = _sub(_ADDRESS, "address", "", out)
    out, n = _PHONE.subn(lambda m: "[PHONE]" if len(re.sub(r"\D", "", m.group(0))) >= 9 else m.group(0), out)
    if "[PHONE]" in out:
        redactions.append("phone")
    out = re.sub(r"^\s*name\s*[:\-].*$", "", out, flags=re.IGNORECASE | re.MULTILINE)

    if name:
        parts = [re.escape(p) for p in name.split() if len(p) > 1]
        if parts:
            out = re.sub(r"\b(" + "|".join(parts) + r")\b", alias, out)
            out = re.sub(rf"(?:{re.escape(alias)}\s*)+", alias + " ", out).replace(alias + " \n", alias + "\n")
            redactions.append("name")

    # Gendered pronouns sometimes appear in third-person summaries.
    pronouns = {"he": "they", "she": "they", "him": "them", "his": "their", "her": "their", "hers": "theirs"}
    out, n = re.subn(r"\b(he|she|him|his|her|hers)\b", lambda m: pronouns[m.group(1).lower()], out, flags=re.IGNORECASE)
    if n:
        redactions.append("pronoun")
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return RedactionResult(out, name, sorted(set(redactions)))


# Terms the compliance agent treats as protected-characteristic references.
PROTECTED_TERMS: dict[str, list[str]] = {
    "age": [r"\bage\b", r"\bhow old\b", r"\byoung\b", r"\bolder\b", r"\bretire", r"\bdate of birth\b", r"\bmillennial", r"\bgen[- ]?z\b"],
    "gender": [r"\bgender\b", r"\bmale\b", r"\bfemale\b", r"\bwoman\b", r"\bman\b", r"\bmaternity\b", r"\bpregnan"],
    "family": [r"\bmarried\b", r"\bmarital\b", r"\bchildren\b", r"\bkids\b", r"\bspouse\b", r"\bfamily plans?\b", r"\bchildcare\b"],
    "origin": [r"\bnationality\b", r"\bethnic", r"\brace\b(?! condition)", r"\bwhere are you from\b", r"\bnative (?:english )?speaker\b", r"\baccent\b", r"\bcitizen"],
    "religion": [r"\breligio", r"\bchurch\b", r"\bmosque\b", r"\btemple\b", r"\bfaith\b"],
    "health": [r"\bdisabilit", r"\bhealth condition", r"\bmedical\b", r"\bsick leave\b", r"\bmental health\b"],
    "orientation": [r"\bsexual orientation\b", r"\bgay\b", r"\blesbian\b"],
}

# Exclusionary or coded wording in job descriptions (flagged, never auto-edited).
JD_BIAS_PATTERNS: dict[str, str] = {
    r"\b(rock ?star|ninja|guru|superhero)\b": "Jargon such as 'rockstar/ninja' can discourage some applicants; describe the skills instead.",
    r"\b(young|youthful|energetic young|recent graduates? only|digital native)\b": "Age-coded wording; describe the experience level required instead.",
    r"\bnative (english )?speaker\b": "'Native speaker' can discriminate by origin; specify the required proficiency level (e.g. 'fluent English').",
    r"\b(he|his|him)\b(?! or)": "Gendered pronoun used for the candidate; prefer 'you' or 'they'.",
    r"\b(manpower|chairman|salesman|guys)\b": "Gendered wording; consider a neutral alternative.",
    r"\b(aggressive|dominant|competitive spirit)\b": "Masculine-coded wording that research links to fewer applicants; consider neutral phrasing.",
    r"\b(must be under|aged? between|maximum age)\b": "Explicit age limit; this is likely unlawful in many jurisdictions.",
    r"\b(culture fit)\b": "'Culture fit' is vague and can mask bias; define the specific values or behaviours.",
}


def find_protected_terms(text: str) -> list[str]:
    hits = []
    for category, patterns in PROTECTED_TERMS.items():
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            hits.append(category)
    return hits


def jd_bias_flags(text: str) -> list[str]:
    flags = []
    for pattern, message in JD_BIAS_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            flags.append(f"\"{m.group(0)}\": {message}")
    return flags
