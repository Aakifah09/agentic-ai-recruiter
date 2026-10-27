import pandas as pd
import streamlit as st

from app_utils import get_store, score_color, workflow_config
from recruiter.orchestrator import rescore
from recruiter.reporting import DECISION_LABELS, shortlist_frame, to_docx, to_json, to_markdown
from recruiter.schemas import EvidenceStrength, HumanReview, Importance, JobProfile, SkillRequirement

store = get_store()
runs = store.list_runs()

st.title("Shortlist & review", anchor=False)
if not runs:
    st.info("No analyses yet. Run one from *New analysis*.", icon=":material/info:")
    st.page_link("app_pages/analyze.py", label="New analysis", icon=":material/person_search:")
    st.stop()

run_ids = [r["run_id"] for r in runs]
if "pending_run_id" in st.session_state:  # set by "Re-score"; applied before the selectbox exists
    st.session_state.current_run_id = st.session_state.pop("pending_run_id")
labels = {r["run_id"]: f"{r['title']} · {r['n_candidates']} candidates · {r['provider']} · {r['created_at'][:16]}" for r in runs}
if st.session_state.current_run_id not in run_ids:
    st.session_state.current_run_id = run_ids[0]
run_id = st.selectbox("Analysis run", run_ids, format_func=labels.get, key="current_run_id")
report = store.load_run(run_id)
reviews = store.reviews(run_id)
by_id = {c.candidate_id: c for c in report.candidates}

st.caption(f":material/gavel: {report.disclaimer}")

# --------------------------------------------------------------------------- KPIs
strong = sum(a.recommendation.name == "STRONG" for a in report.assessments)
reviewed = sum(r.decision != "pending" for r in reviews.values())
with st.container(horizontal=True):
    st.metric("Candidates", len(report.assessments), border=True)
    st.metric("Strong evidence matches", strong, border=True)
    st.metric("Top score", f"{report.assessments[0].overall_score:.0f}", border=True)
    st.metric("Human-reviewed", f"{reviewed}/{len(report.assessments)}", border=True)
    st.metric("Compliance issues", len(report.compliance.issues), border=True)

tab_list, tab_detail, tab_job, tab_compliance, tab_trace, tab_export = st.tabs(
    [":material/leaderboard: Shortlist", ":material/person: Candidate review", ":material/work: Job profile",
     ":material/shield: Compliance", ":material/timeline: Agent trace", ":material/download: Export"]
)

# --------------------------------------------------------------------------- shortlist
with tab_list:
    df = shortlist_frame(report, reviews)
    st.dataframe(
        df,
        hide_index=True,
        column_order=["rank", "candidate", "overall_score", "essential_coverage", "years_experience",
                      "confidence", "ai_recommendation", "recruiter_decision", "demonstrated_skills", "missing_essentials"],
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "candidate": "Candidate",
            "overall_score": st.column_config.ProgressColumn("Match score", min_value=0, max_value=100, format="%.0f"),
            "essential_coverage": st.column_config.ProgressColumn("Essential coverage", min_value=0, max_value=100, format="%.0f%%"),
            "years_experience": st.column_config.NumberColumn("Years exp.", format="%.1f"),
            "confidence": "Confidence",
            "ai_recommendation": "AI suggestion",
            "recruiter_decision": "Recruiter decision",
            "demonstrated_skills": "Demonstrated skills",
            "missing_essentials": "Missing essentials",
        },
    )
    chart = df.melt(id_vars="candidate", value_vars=["skill_score", "experience_score", "qualification_score"],
                    var_name="component", value_name="score")
    chart["component"] = chart["component"].str.replace("_score", "").str.title()
    st.subheader("Score components", anchor=False)
    st.bar_chart(chart, x="candidate", y="score", color="component", stack=False, sort=False, height=320)

# --------------------------------------------------------------------------- candidate detail + human review
with tab_detail:
    names = {a.candidate_id: f"#{a.rank} {a.display_name} ({a.overall_score:.0f})" for a in report.assessments}
    cid = st.selectbox("Candidate", list(names), format_func=names.get, key=f"cand_{run_id}")
    a = next(x for x in report.assessments if x.candidate_id == cid)
    c = by_id[cid]

    col_main, col_review = st.columns([2, 1], gap="large")
    with col_main:
        with st.container(horizontal=True):
            st.badge(a.recommendation.value, color=score_color(a.overall_score))
            st.badge(f"Confidence: {a.confidence}", color="gray", icon=":material/speed:")
            st.badge(f"{a.relevant_years:g} relevant yrs / {c.total_years_experience:g} total", color="blue",
                     icon=":material/schedule:")
        st.markdown(a.rationale)
        s_col, g_col = st.columns(2)
        with s_col:
            st.markdown("**Strengths**")
            for s in a.strengths or ["-"]:
                st.markdown(f"- {s}")
        with g_col:
            st.markdown("**Gaps / to explore**")
            for g in a.gaps or ["-"]:
                st.markdown(f"- {g}")

        st.markdown("**Evidence by requirement**")
        ev = pd.DataFrame([{
            "Requirement": m.skill, "Importance": m.importance.value, "Weight": m.weight,
            "Evidence": m.strength.value, "Credit": m.credit, "Excerpt": m.evidence[0] if m.evidence else "",
        } for m in a.requirement_matches])
        st.dataframe(ev, hide_index=True, column_config={
            "Weight": st.column_config.NumberColumn(format="percent"),
            "Credit": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Excerpt": st.column_config.TextColumn(width="large"),
        })

        plan = report.interview_plans.get(cid)
        st.markdown("**Interview plan**")
        if plan and plan.questions:
            for i, q in enumerate(plan.questions, 1):
                with st.expander(f"{i}. {q.question}", icon=":material/quiz:"):
                    st.caption(f"Purpose: {q.purpose.replace('_', ' ')}" + (f" · skill: {q.target_skill}" if q.target_skill else ""))
                    st.markdown(f"**A strong answer:** {q.what_good_looks_like}")
        else:
            st.caption("No interview plan generated (outside the top-k). Increase *Interview plans for top-k* and re-run.")

        with st.expander("Extracted profile", icon=":material/badge:"):
            st.caption(f"Source: {c.source_file} · Redacted before analysis: {', '.join(c.redactions) or 'nothing'}")
            for w in c.parse_warnings:
                st.warning(w, icon=":material/warning:")
            if c.experience:
                st.dataframe(pd.DataFrame([{"Role": e.title, "Organisation": e.organisation, "From": e.start,
                                            "To": e.end, "Months": e.months} for e in c.experience]), hide_index=True)
            st.dataframe(pd.DataFrame([{"Skill": s.skill, "Category": s.category, "Evidence": s.strength.value,
                                        "Excerpt": s.snippets[0] if s.snippets else ""} for s in c.skills]), hide_index=True)
            st.caption("Education: " + ("; ".join(c.education) or "-"))

    with col_review, st.container(border=True):
        st.subheader(":material/how_to_reg: Recruiter review", anchor=False)
        current = reviews.get(cid, HumanReview(candidate_id=cid))
        with st.form(f"review_{run_id}_{cid}", border=False):
            decision = st.segmented_control("Decision", list(DECISION_LABELS), default=current.decision,
                                            format_func=DECISION_LABELS.get, required=True)
            reviewer = st.text_input("Reviewer", value=current.reviewer)
            notes = st.text_area("Notes", value=current.notes, height=140,
                                 placeholder="Reasoning, follow-ups, interview feedback...")
            if st.form_submit_button("Save review", type="primary", icon=":material/save:"):
                store.save_review(run_id, HumanReview(candidate_id=cid, decision=decision, reviewer=reviewer, notes=notes))
                st.toast("Review saved", icon=":material/check:")
                st.rerun()
        st.caption("The AI suggestion is advisory. Every progression decision is recorded here with an audit trail.")
        audit = [r for r in store.audit_log(run_id) if r["candidate_id"] == cid]
        if audit:
            with st.expander("Audit trail", icon=":material/history:"):
                for r in reversed(audit):
                    st.caption(f"{r['at'][:19]} · {r['reviewer'] or 'recruiter'} → {DECISION_LABELS[r['decision']]}")

# --------------------------------------------------------------------------- job profile (editable)
with tab_job:
    job = report.job
    st.markdown(f"**{job.title}** · seniority: {job.seniority} · min experience: "
                f"{job.min_years_experience if job.min_years_experience is not None else 'not stated'} yrs · "
                f"education: {', '.join(job.education_requirements) or 'not stated'}")
    if job.responsibilities:
        with st.expander("Responsibilities", icon=":material/checklist:"):
            for r in job.responsibilities:
                st.markdown(f"- {r}")
    st.caption("Correct the extracted requirements, then re-score. Re-scoring reuses the parsed resumes and runs instantly.")
    skills_df = pd.DataFrame([{
        "skill": s.skill, "alternatives": ", ".join(s.alternatives), "importance": s.importance.value,
        "weight": round(s.weight * 100, 1), "rationale": s.rationale,
    } for s in job.skills])
    edited = st.data_editor(
        skills_df, num_rows="dynamic", hide_index=True, key=f"job_editor_{run_id}",
        column_config={
            "skill": st.column_config.TextColumn("Skill", required=True),
            "alternatives": st.column_config.TextColumn("Accepted alternatives", help="Comma separated"),
            "importance": st.column_config.SelectboxColumn("Importance", options=["essential", "desirable"], required=True),
            "weight": st.column_config.NumberColumn("Relative weight", min_value=0, max_value=100, step=0.5,
                                                    help="Re-normalised to 100% on re-score"),
            "rationale": st.column_config.TextColumn("Why", disabled=True),
        },
    )
    min_years = st.number_input("Minimum years of experience", 0.0, 40.0,
                                float(job.min_years_experience or 0), 0.5, key=f"min_years_{run_id}")
    if st.button("Re-score with edited profile", icon=":material/refresh:", type="primary"):
        rows = edited.dropna(subset=["skill"])
        new_skills = [SkillRequirement(
            skill=str(r.skill).strip(), importance=Importance(r.importance),
            weight=float(r.weight or 0) / 100,
            alternatives=[x.strip() for x in str(r.alternatives or "").split(",") if x.strip()],
            rationale=str(r.rationale or "edited by recruiter"),
        ) for r in rows.itertuples() if str(r.skill).strip()]
        total = sum(s.weight for s in new_skills) or 1
        for s in new_skills:
            s.weight = round(s.weight / total, 4)
        new_job = JobProfile(**{**job.model_dump(), "skills": [s.model_dump() for s in new_skills],
                                "min_years_experience": min_years or None})
        cfg = workflow_config()
        new_report = rescore(report, new_job, cfg.scoring, cfg.interview_top_k)
        job_row = next(r for r in runs if r["run_id"] == run_id)
        store.save_run(new_report, f"{job_row['title']} (edited {new_report.run_id})")
        st.session_state.pending_run_id = new_report.run_id
        st.rerun()

# --------------------------------------------------------------------------- compliance
with tab_compliance:
    if report.compliance.passed and not report.compliance.issues:
        st.success("No fairness, privacy or grounding issues detected.", icon=":material/verified_user:")
    for issue in report.compliance.issues:
        fn = {"blocked": st.error, "warning": st.warning, "info": st.info}[issue.severity]
        fn(f"**{issue.location}** - {issue.message}", icon=":material/shield:")
    st.markdown(
        "**Safeguards applied to every run**\n"
        "- Names, contact details, dates of birth, nationality, gender, marital status and similar lines are removed before any agent sees a resume.\n"
        "- Gendered pronouns are neutralised; blind-review aliases hide names in the UI.\n"
        "- Evidence excerpts must appear verbatim in the resume, or they are discarded.\n"
        "- Interview questions and narrative text touching protected characteristics are removed.\n"
        "- The system only suggests; it cannot reject. Decisions are made and logged by a recruiter."
    )

# --------------------------------------------------------------------------- trace
with tab_trace:
    trace_df = pd.DataFrame([s.model_dump() for s in report.trace])
    st.dataframe(trace_df[["agent", "mode", "status", "duration_ms", "detail"]], hide_index=True,
                 column_config={"duration_ms": st.column_config.NumberColumn("Duration (ms)", format="%.0f"),
                                "detail": st.column_config.TextColumn(width="large")})
    st.caption(f"Provider: {report.provider} · fallback steps: {sum(s.status == 'fallback' for s in report.trace)}")

# --------------------------------------------------------------------------- export
with tab_export:
    stem = f"shortlist_{report.job.title.lower().replace(' ', '_')[:40]}_{run_id}"
    with st.container(horizontal=True):
        st.download_button("Markdown report", to_markdown(report, reviews), f"{stem}.md", "text/markdown",
                           icon=":material/article:")
        st.download_button("Word report", to_docx(report, reviews), f"{stem}.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           icon=":material/description:")
        st.download_button("Shortlist CSV", shortlist_frame(report, reviews).to_csv(index=False), f"{stem}.csv",
                           "text/csv", icon=":material/table:")
        st.download_button("Full JSON", to_json(report, reviews), f"{stem}.json", "application/json",
                           icon=":material/data_object:")
    with st.expander("Preview Markdown report", icon=":material/visibility:"):
        st.markdown(to_markdown(report, reviews))
