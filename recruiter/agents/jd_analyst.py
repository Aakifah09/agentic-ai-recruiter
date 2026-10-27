"""Agent 1 - Job Description Analyst.

Turns free-text job descriptions into a ``JobProfile`` with essential vs
desirable skills and explicit, normalised importance weights.
"""

from __future__ import annotations

import re

from recruiter import prompts
from recruiter.agents.base import Agent
from recruiter.llm import structured_call
from recruiter.privacy import jd_bias_flags
from recruiter.schemas import Importance, JobProfile, SkillRequirement
from recruiter.skills import DEGREE_RANK, TAXONOMY, category_of, find_certifications, find_degrees, find_skills, normalise_skill
from recruiter.textutils import split_sections, stated_years

# Base importance used before normalisation.
ESSENTIAL_BASE = 1.0
DESIRABLE_BASE = 0.4
SOFT_SKILL_FACTOR = 0.6  # soft skills are hard to evidence from a resume, so weigh them lower
INLINE_PREFERRED = re.compile(r"\b(preferred|nice to have|nice-to-have|a plus|bonus|desirable|advantage|ideally|familiarity)\b", re.I)
INLINE_REQUIRED = re.compile(r"\b(must|required|essential|strong|proven|proficien|expert|solid)\b", re.I)


class JobDescriptionAnalyst(Agent):
    name = "Job Description Analyst"
    description = "Extracts role, responsibilities, essential/desirable skills and importance weights."

    def run(self, job_text: str) -> JobProfile:
        profile = self._execute(
            (lambda: self._run_llm(job_text)) if self.provider else None,
            lambda: self._run_rules(job_text),
        )
        profile.bias_flags = jd_bias_flags(job_text)
        normalise_weights(profile.skills)
        if self.last_step:
            self.last_step.detail = (
                f"{len(profile.essential_skills)} essential / {len(profile.desirable_skills)} desirable skills; "
                f"min years: {profile.min_years_experience}; bias flags: {len(profile.bias_flags)}"
                + (f" | {self.last_step.detail}" if self.last_step.detail else "")
            )
        return profile

    # ------------------------------------------------------------------ rules
    def _run_rules(self, text: str) -> JobProfile:
        sections = split_sections(text)
        title = extract_title(text)
        raw: dict[str, dict] = {}
        standalone: set[str] = set()
        alt_groups: list[list[str]] = []

        def add(skill: str, importance: Importance, why: str, boost: float = 0.0) -> None:
            entry = raw.setdefault(skill, {"importance": importance, "score": 0.0, "why": [], "mentions": 0})
            if importance == Importance.ESSENTIAL:
                entry["importance"] = Importance.ESSENTIAL
            entry["mentions"] += 1
            entry["score"] = max(entry["score"], (ESSENTIAL_BASE if importance == Importance.ESSENTIAL else DESIRABLE_BASE) + boost)
            if why not in entry["why"]:
                entry["why"].append(why)

        for section, lines in sections.items():
            if section in {"benefits", "about"}:
                continue
            for line in lines:
                hits = find_skills(line)
                if not hits:
                    continue
                if section == "preferred" or (section != "required" and INLINE_PREFERRED.search(line)):
                    importance, why = Importance.DESIRABLE, "listed as preferred"
                elif section == "required":
                    if INLINE_PREFERRED.search(line):
                        importance, why = Importance.DESIRABLE, "qualified as a plus within requirements"
                    else:
                        importance, why = Importance.ESSENTIAL, "listed under requirements"
                elif section == "responsibilities":
                    importance, why = Importance.DESIRABLE, "mentioned in responsibilities"
                    if INLINE_REQUIRED.search(line):
                        importance, why = Importance.ESSENTIAL, "stated as required in responsibilities"
                else:
                    importance, why = Importance.DESIRABLE, "mentioned in role description"
                groups = alternative_groups(line, hits)
                grouped = {s for g in groups for s in g}
                for skill in hits:
                    add(skill, importance, why)
                    if skill not in grouped:
                        standalone.add(skill)
                alt_groups.extend(groups)

        # Skills in the title are central to the role.
        for skill in find_skills(title):
            add(skill, Importance.ESSENTIAL, "appears in job title", boost=0.5)

        # "X or Y": fold alternatives into the first skill unless they are also required on their own.
        alternatives: dict[str, list[str]] = {}
        for group in alt_groups:
            primary, *rest = group
            for alt in rest:
                if alt in raw and alt not in standalone and primary in raw:
                    alternatives.setdefault(primary, []).append(alt)
                    raw[primary]["score"] = max(raw[primary]["score"], raw[alt]["score"])
                    del raw[alt]

        # Repeated mentions raise importance (bounded).
        all_hits = find_skills(text)
        skills = []
        for skill, entry in raw.items():
            repeat_boost = min(0.3, 0.1 * (len(all_hits.get(skill, [])) - 1))
            score = entry["score"] + repeat_boost
            if category_of(skill) == "soft":
                score *= SOFT_SKILL_FACTOR
            skills.append(
                SkillRequirement(
                    skill=skill,
                    category=category_of(skill),
                    importance=entry["importance"],
                    weight=min(1.0, score / 2),  # provisional, normalised later
                    alternatives=alternatives.get(skill, []),
                    rationale="; ".join(entry["why"]),
                )
            )

        required_text = "\n".join(sections.get("required", [])) or text
        years = stated_years(required_text) or stated_years(text)
        degrees = find_degrees(required_text)
        return JobProfile(
            title=title,
            summary=" ".join(sections.get("header", [])[1:4])[:400],
            responsibilities=[l.lstrip("- ").strip() for l in sections.get("responsibilities", [])][:10],
            skills=skills,
            min_years_experience=min(years) if years else None,
            education_requirements=[min(degrees, key=DEGREE_RANK.get)] if degrees else [],
            certifications=find_certifications(text),
            seniority=infer_seniority(title, min(years) if years else None),
        )

    # -------------------------------------------------------------------- llm
    def _run_llm(self, text: str) -> JobProfile:
        result: prompts.LLMJobExtraction = structured_call(
            self.provider, prompts.JD_SYSTEM, prompts.JD_USER.format(job_text=text[:12000]), prompts.LLMJobExtraction
        )
        skills: dict[str, SkillRequirement] = {}
        for items, importance, base in (
            (result.required_skills, Importance.ESSENTIAL, 1.0),
            (result.preferred_skills, Importance.DESIRABLE, 0.5),
        ):
            for item in items:
                name, alts = split_alternatives(item.skill)
                if not name or len(name) > 60 or name in skills:
                    continue
                weight = base * item.priority / 5
                if category_of(name) == "soft":
                    weight *= SOFT_SKILL_FACTOR
                skills[name] = SkillRequirement(
                    skill=name, category=category_of(name), importance=importance, weight=min(1.0, weight),
                    alternatives=alts,
                    rationale=f"LLM: {'required' if importance == Importance.ESSENTIAL else 'preferred'}, priority {item.priority}/5",
                )

        # Hybrid reconciliation. Where the JD's own structure (Requirements vs Nice to have)
        # classifies a skill, that wins over the LLM - small models often blur the two. The LLM
        # contributes open-vocabulary skills the taxonomy lacks and its priority judgement.
        rules_profile = self._run_rules(text)
        rules_by_skill = {a: req for req in rules_profile.skills for a in req.accepted()}
        for name, req in list(skills.items()):
            ruled = rules_by_skill.get(name)
            if ruled is None:
                continue
            if ruled.skill != name and ruled.skill in skills:  # alternative already represented
                del skills[name]
                continue
            req.importance = ruled.importance
            req.alternatives = ruled.alternatives
            req.weight = round((req.weight + ruled.weight) / 2, 4)
            req.rationale = f"{ruled.rationale}; {req.rationale}"
            if ruled.skill != name:
                skills[ruled.skill] = req.model_copy(update={"skill": ruled.skill})
                del skills[name]
        covered = {a for req in skills.values() for a in req.accepted()}
        for req in rules_profile.skills:
            if req.skill not in covered and req.importance == Importance.ESSENTIAL:
                req.rationale += " (added by rule-based cross-check)"
                skills[req.skill] = req

        degrees = find_degrees(" ".join(result.education))
        return JobProfile(
            title=result.title.strip() or rules_profile.title,
            summary=result.summary[:400],
            responsibilities=[r.strip() for r in result.responsibilities if r.strip()][:10] or rules_profile.responsibilities,
            skills=list(skills.values()),
            min_years_experience=result.min_years_experience if result.min_years_experience is not None else rules_profile.min_years_experience,
            education_requirements=[min(degrees, key=DEGREE_RANK.get)] if degrees else rules_profile.education_requirements,
            certifications=result.certifications or rules_profile.certifications,
            seniority=result.seniority if result.seniority != "unspecified" else rules_profile.seniority,
        )


def normalise_weights(skills: list[SkillRequirement]) -> None:
    total = sum(s.weight for s in skills)
    if total <= 0:
        for s in skills:
            s.weight = round(1 / len(skills), 4) if skills else 0
        return
    for s in skills:
        s.weight = round(s.weight / total, 4)
    skills.sort(key=lambda s: (s.importance != Importance.ESSENTIAL, -s.weight, s.skill))


def extract_title(text: str) -> str:
    m = re.search(r"^\s*(?:job\s*title|position|role|title)\s*[:\-]\s*(.+)$", text, re.I | re.M)
    if m:
        return m.group(1).strip()[:120]
    for line in text.splitlines():
        line = line.strip().strip("#*").strip()
        if line:
            return line[:120]
    return "Untitled role"


def infer_seniority(title: str, years: float | None) -> str:
    t = title.lower()
    for words, level in (
        (("intern", "internship", "trainee", "placement"), "intern"),
        (("junior", "graduate", "entry", "associate"), "junior"),
        (("lead", "principal", "head of", "staff", "manager"), "lead"),
        (("senior", "sr.", "sr "), "senior"),
    ):
        if any(w in t for w in words):
            return level
    if years is None:
        return "unspecified"
    return "junior" if years < 2 else "mid" if years < 5 else "senior"


_ALT_GAP = re.compile(r"^\s*(?:or|/|and/or|either)\s*$", re.I)


def alternative_groups(line: str, hits: dict[str, list[tuple[int, int]]]) -> list[list[str]]:
    """Chains of skills joined by "or" / "/" in one line, e.g. "Power BI or Tableau"."""
    spans = sorted((start, end, skill) for skill, ss in hits.items() for start, end in ss)
    groups: list[list[str]] = []
    current: list[str] = []
    for (s1, e1, a), (s2, e2, b) in zip(spans, spans[1:]):
        if a != b and _ALT_GAP.match(line[e1:s2]):
            if not current:
                current = [a]
            if b not in current:
                current.append(b)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def split_alternatives(raw: str) -> tuple[str, list[str]]:
    """Normalise an LLM skill string, splitting "FastAPI or Flask" into primary + alternatives."""
    whole = normalise_skill(raw)
    if whole in TAXONOMY or not re.search(r"\s(?:or)\s|/", raw):
        return whole, []
    parts = [normalise_skill(p) for p in re.split(r"\s+or\s+|\s*/\s*", raw) if p.strip()]
    parts = list(dict.fromkeys(p for p in parts if p))
    return (parts[0], parts[1:]) if parts else (whole, [])
