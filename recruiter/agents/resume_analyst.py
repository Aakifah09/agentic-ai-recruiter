"""Agent 2 - Resume Analyst.

Redacts personal data, then extracts skills (with verbatim evidence and an
evidence-strength label), work history, total experience, education and
certifications. In LLM mode every quoted piece of evidence is checked against
the source text; ungrounded claims are discarded.
"""

from __future__ import annotations

import re

from recruiter import prompts
from recruiter.agents.base import Agent
from recruiter.llm import structured_call
from recruiter.privacy import redact
from recruiter.schemas import CandidateProfile, EvidenceStrength, ExperienceEntry, JobProfile, SkillEvidence
from recruiter.skills import IMPLIED_SKILLS, category_of, find_certifications, find_degrees, find_skills, normalise_skill
from recruiter.textutils import (
    DATE_RANGE,
    is_grounded,
    merged_months,
    months_between,
    parse_point,
    sentence_around,
    split_sections,
    stated_years,
)

WORK_SECTIONS = {"experience", "projects"}
ACTION_VERBS = re.compile(
    r"\b(built|developed|designed|implemented|deployed|led|created|engineered|automated|optimi[sz]ed|migrated|"
    r"architected|delivered|trained|fine-tuned|integrated|launched|maintained|analy[sz]ed|reduced|improved|wrote|"
    r"used|applied|managed|owned|shipped|productioni[sz]ed|scaled|tested|refactored|modelled|modeled)\b",
    re.I,
)


class ResumeAnalyst(Agent):
    name = "Resume Analyst"
    description = "Redacts PII, extracts skills with verbatim evidence, experience timeline and qualifications."

    def run(self, raw_text: str, candidate_id: str, source_file: str = "", alias: str | None = None,
            job: JobProfile | None = None) -> CandidateProfile:
        red = redact(raw_text)
        display = alias or red.name or candidate_id
        profile = self._execute(
            (lambda: self._run_llm(red.text, candidate_id, job)) if self.provider else None,
            lambda: self._run_rules(red.text, candidate_id),
        )
        profile.display_name = display
        profile.source_file = source_file
        profile.redactions = red.redactions
        if len(red.text) < 300:
            profile.parse_warnings.append("Very little text extracted; results may be incomplete.")
        if self.last_step:
            self.last_step.agent = f"{self.name} [{display}]"
            self.last_step.detail = (
                f"{len(profile.skills)} skills, {profile.total_years_experience} yrs, redacted: {', '.join(red.redactions) or 'none'}"
                + (f" | {self.last_step.detail}" if self.last_step.detail else "")
            )
        return profile

    # ------------------------------------------------------------------ rules
    def _run_rules(self, text: str, candidate_id: str) -> CandidateProfile:
        sections = split_sections(text)
        section_of_line: list[tuple[str, str]] = [(sec, line) for sec, lines in sections.items() for line in lines]
        work_text = "\n".join(line for sec, line in section_of_line if sec in WORK_SECTIONS)

        skills = []
        for skill, spans in find_skills(text).items():
            snippets_work, snippets_other = [], []
            for start, end in spans:
                snippet = sentence_around(text, start, end)
                in_work = snippet and (snippet in work_text or ACTION_VERBS.search(snippet))
                target = snippets_work if in_work else snippets_other
                if snippet and snippet not in target:
                    target.append(snippet)
            strength = EvidenceStrength.DEMONSTRATED if snippets_work else EvidenceStrength.LISTED
            skills.append(SkillEvidence(
                skill=skill, category=category_of(skill), strength=strength,
                snippets=(snippets_work + snippets_other)[:3],
            ))

        have = {s.skill for s in skills}
        for s in [s for s in skills if s.strength == EvidenceStrength.DEMONSTRATED]:
            implied = IMPLIED_SKILLS.get(s.skill)
            if implied and implied not in have:
                skills.append(SkillEvidence(skill=implied, category=category_of(implied),
                                            strength=EvidenceStrength.DEMONSTRATED, snippets=s.snippets[:1]))
                have.add(implied)

        experience, intervals = parse_experience(sections.get("experience", []) + sections.get("header", []))
        months = merged_months(intervals)
        years = round(months / 12, 1)
        warnings = []
        if not intervals:
            stated = stated_years("\n".join(sections.get("summary", []) + sections.get("header", [])))
            if stated:
                years = max(stated)
            else:
                warnings.append("No dated work history found; experience could not be measured.")

        education_lines = sections.get("education", [])
        return CandidateProfile(
            candidate_id=candidate_id,
            display_name=candidate_id,
            headline=headline(sections),
            total_years_experience=years,
            skills=sorted(skills, key=lambda s: (s.strength != EvidenceStrength.DEMONSTRATED, s.skill)),
            experience=experience,
            education=education_lines[:6] or find_degrees(text),
            certifications=list(dict.fromkeys(
                find_certifications(text) + [l for l in sections.get("certifications", []) if len(l) < 120][:6])),
            parse_warnings=warnings,
        )

    # -------------------------------------------------------------------- llm
    def _run_llm(self, text: str, candidate_id: str, job: JobProfile | None) -> CandidateProfile:
        focus = ", ".join(s.skill for s in job.skills) if job else "any"
        result: prompts.LLMResumeExtraction = structured_call(
            self.provider, prompts.RESUME_SYSTEM,
            prompts.RESUME_USER.format(resume_text=text[:12000], focus_skills=focus),
            prompts.LLMResumeExtraction,
        )
        base = self._run_rules(text, candidate_id)
        merged = {s.skill: s for s in base.skills}
        rejected = 0
        for item in result.skills:
            name = normalise_skill(item.skill)
            if not name or not is_grounded(item.evidence_quote, text):
                rejected += 1
                continue
            strength = EvidenceStrength.DEMONSTRATED if item.used_in_work else EvidenceStrength.LISTED
            existing = merged.get(name)
            if existing is None:
                merged[name] = SkillEvidence(skill=name, category=category_of(name), strength=strength,
                                             snippets=[item.evidence_quote.strip()])
            else:
                if strength == EvidenceStrength.DEMONSTRATED:
                    existing.strength = strength
                if item.evidence_quote.strip() not in existing.snippets:
                    existing.snippets.insert(0, item.evidence_quote.strip())
                    existing.snippets = existing.snippets[:3]

        if not base.experience and result.experience:
            entries, intervals = [], []
            for e in result.experience:
                s, en = parse_point(e.start or ""), parse_point(e.end or "")
                months = months_between(s, en) if s and en else 0
                if s and en:
                    intervals.append((s, en))
                entries.append(ExperienceEntry(title=e.title, organisation=e.organisation, start=e.start, end=e.end,
                                               months=months, highlights=e.highlights[:3]))
            base.experience = entries
            if intervals:
                base.total_years_experience = round(merged_months(intervals) / 12, 1)
                base.parse_warnings = [w for w in base.parse_warnings if "experience could not" not in w]
            elif result.total_years_experience:
                base.total_years_experience = float(result.total_years_experience)

        base.skills = sorted(merged.values(), key=lambda s: (s.strength != EvidenceStrength.DEMONSTRATED, s.skill))
        base.headline = base.headline or result.headline
        if rejected:
            base.parse_warnings.append(f"{rejected} LLM skill claim(s) discarded: evidence not found in resume.")
        return base


def parse_experience(lines: list[str]) -> tuple[list[ExperienceEntry], list]:
    """Find dated roles. The line with the date range (or the one above it) is the role title."""
    entries: list[ExperienceEntry] = []
    intervals = []
    for i, line in enumerate(lines):
        m = DATE_RANGE.search(line)
        if not m:
            continue
        start, end = parse_point(m.group("start")), parse_point(m.group("end"))
        if not start or not end or end < start:
            continue
        label = (line[: m.start()] + line[m.end():]).strip(" |,-–—()")
        if len(label) < 3 and i > 0:
            label = lines[i - 1]
        parts = re.split(r"\s+(?:at|@)\s+|\s*[|,–—]\s*", label)
        title = parts[0].strip() if parts else label
        org = parts[1].strip() if len(parts) > 1 else ""
        highlights = []
        for nxt in lines[i + 1 : i + 6]:
            if DATE_RANGE.search(nxt):
                break
            if nxt.startswith("- ") or ACTION_VERBS.search(nxt):
                highlights.append(nxt.lstrip("- ").strip())
        entries.append(ExperienceEntry(title=title[:100], organisation=org[:100], start=m.group("start"),
                                       end=m.group("end"), months=months_between(start, end),
                                       highlights=highlights[:3]))
        intervals.append((start, end))
    return entries, intervals


def headline(sections: dict[str, list[str]]) -> str:
    summary = sections.get("summary", [])
    if summary:
        return summary[0][:200]
    header = [l for l in sections.get("header", []) if "[" not in l]
    return header[0][:200] if header else ""
