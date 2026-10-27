from pathlib import Path

import streamlit as st

from app_utils import STAGE_LABELS, get_store, workflow_config
from recruiter.documents import DocumentError, extract_file, extract_text
from recruiter.orchestrator import RecruitmentWorkflow, ResumeInput

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
SAMPLE_JOBS = {p.stem.replace("_", " ").title(): p for p in sorted((SAMPLES / "jobs").glob("*.txt"))}

st.title("New analysis", anchor=False)
st.caption(
    "Compare resumes against a job description. Five specialised agents analyse the role, read each resume, "
    "match evidence, prepare interview questions and run a fairness check. "
    "**Output supports a recruiter's decision; it never makes one.**"
)

left, right = st.columns(2, gap="large")

# --------------------------------------------------------------------------- job description
with left, st.container(border=True):
    st.subheader(":material/work: Job description", anchor=False)
    source = st.segmented_control("Source", ["Paste", "Upload", "Sample"], default="Sample", key="jd_source",
                                  label_visibility="collapsed")
    job_text = ""
    if source == "Paste":
        job_text = st.text_area("Job description text", height=320, key="jd_text",
                                placeholder="Paste the full job description...")
    elif source == "Upload":
        f = st.file_uploader("Job description file", type=["pdf", "docx", "txt", "md"], key="jd_file")
        if f:
            try:
                job_text = extract_text(f.getvalue(), f.name)
            except DocumentError as exc:
                st.error(str(exc), icon=":material/error:")
    else:
        choice = st.selectbox("Sample job", list(SAMPLE_JOBS), key="jd_sample")
        job_text = extract_file(SAMPLE_JOBS[choice])
    if job_text and source != "Paste":
        with st.expander("Preview", icon=":material/visibility:"):
            st.text(job_text)

# --------------------------------------------------------------------------- resumes
with right, st.container(border=True):
    st.subheader(":material/description: Resumes", anchor=False)
    files = st.file_uploader("Upload resumes (PDF, DOCX or TXT)", type=["pdf", "docx", "txt", "md"],
                             accept_multiple_files=True, key="resume_files")
    use_samples = st.toggle("Include the 11 fictional sample resumes", value=not files, key="use_samples")

    resumes: list[ResumeInput] = []
    errors = []
    for f in files or []:
        try:
            resumes.append(ResumeInput(f.name, extract_text(f.getvalue(), f.name)))
        except DocumentError as exc:
            errors.append(str(exc))
    if use_samples:
        for p in sorted((SAMPLES / "resumes").glob("*.txt")):
            resumes.append(ResumeInput(p.name, extract_file(p)))
    for e in errors:
        st.warning(e, icon=":material/warning:")
    st.metric("Resumes ready", len(resumes))
    if resumes:
        with st.expander("Files", icon=":material/folder_open:"):
            for r in resumes:
                st.caption(f"{r.filename} · {len(r.text):,} characters")

# --------------------------------------------------------------------------- run
config = workflow_config()
llm_on = config.provider_spec != "rules" and config.llm_stages
if llm_on:
    n_calls = ("jd" in config.llm_stages) + len(resumes) * (("resume" in config.llm_stages) + ("narrative" in config.llm_stages)) \
        + min(config.interview_top_k, len(resumes)) * ("interview" in config.llm_stages)
    st.info(
        f"Using **{config.provider_spec}** for: {', '.join(STAGE_LABELS[s] for s in config.llm_stages)}. "
        f"About {n_calls} LLM call(s); on a CPU this can take {n_calls}-{n_calls * 2} minutes. "
        "Any failed call falls back to the rule-based agent automatically.",
        icon=":material/memory:",
    )

if st.button("Run agent workflow", type="primary", icon=":material/play_arrow:",
             disabled=not (job_text.strip() and resumes)):
    workflow = RecruitmentWorkflow(config)
    with st.status("Running agents...", expanded=True) as status:
        bar = st.progress(0.0)
        log = st.empty()

        def progress(msg: str, frac: float) -> None:
            bar.progress(min(1.0, frac))
            log.caption(msg)

        try:
            report = workflow.run(job_text, resumes, progress=progress)
        except Exception as exc:  # surface any unexpected failure in the UI
            status.update(label="Workflow failed", state="error")
            st.exception(exc)
            st.stop()
        get_store().save_run(report, job_text, {f"C{i + 1:03d}": r.text for i, r in enumerate(resumes)})
        st.session_state.current_run_id = report.run_id
        status.update(label=f"Analysed {len(report.candidates)} candidates in "
                            f"{report.trace[-1].duration_ms / 1000:.1f}s", state="complete", expanded=False)

    for step in report.trace:
        icon = {"ok": ":material/check_circle:", "fallback": ":material/swap_horiz:", "error": ":material/error:"}[step.status]
        with st.expander(f"{step.agent} · {step.mode} · {step.duration_ms / 1000:.2f}s", icon=icon, type="step"):
            st.caption(step.detail or "-")

    top = report.assessments[0]
    st.success(f"Top evidence match: **{top.display_name}** ({top.overall_score:.0f}/100). "
               "Open *Shortlist & review* to inspect evidence and record decisions.", icon=":material/task_alt:")
    st.page_link("app_pages/shortlist.py", label="Go to shortlist", icon=":material/arrow_forward:")
