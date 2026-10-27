import pandas as pd
import streamlit as st

from app_utils import get_store, workflow_config
from recruiter.evaluation import run_evaluation

store = get_store()

st.title("Evaluation", anchor=False)
st.caption(
    "Measures each agent against hand-labelled ground truth for 3 job descriptions and 11 fictional resumes "
    "(`data/eval/gold.json`). Every run is logged as an experiment in SQLite so prompt, model and scoring-policy "
    "changes can be compared."
)

METRIC_GROUPS = {
    "Extraction": ["jd_skill_f1", "jd_importance_accuracy", "resume_skill_precision", "resume_skill_recall", "resume_skill_f1"],
    "Ranking": ["ndcg@3", "ndcg@5", "precision@3", "spearman_rho", "pairwise_accuracy"],
    "Responsible AI": ["evidence_grounding_rate", "pii_leak_rate", "bias_flag_recall",
                       "counterfactual_max_score_shift", "counterfactual_mean_score_shift"],
    "Efficiency": ["latency_s_per_candidate"],
}
HELP = {
    "jd_skill_f1": "F1 of skills extracted from job descriptions vs gold requirements",
    "jd_importance_accuracy": "Share of gold skills labelled correctly as essential vs desirable",
    "resume_skill_f1": "Micro F1 of skills extracted from resumes",
    "ndcg@3": "Ranking quality of the top 3 against graded relevance (1.0 = ideal order)",
    "precision@3": "Share of top-3 candidates with gold grade >= 2",
    "spearman_rho": "Rank correlation between scores and gold grades",
    "pairwise_accuracy": "Share of differently-graded candidate pairs ordered correctly",
    "evidence_grounding_rate": "Share of evidence excerpts found verbatim in the source resume",
    "pii_leak_rate": "Share of candidate emails appearing anywhere in the output (lower is better)",
    "bias_flag_recall": "Share of planted exclusionary JD phrases that were flagged",
    "counterfactual_max_score_shift": "Largest score change when only name / gender / nationality / DOB change (0 = invariant)",
    "latency_s_per_candidate": "Wall-clock seconds per candidate for the whole workflow",
}

cfg = workflow_config()
with st.container(border=True):
    st.markdown(f"**Current configuration:** provider `{cfg.provider_spec}`"
                + (f", LLM stages: {', '.join(cfg.llm_stages)}" if cfg.provider_spec != "rules" else ""))
    name = st.text_input("Experiment name", value=f"{cfg.provider_spec}-listed{cfg.scoring.listed_credit:.2f}")
    fairness = st.toggle("Include counterfactual fairness test", value=True,
                         help="Re-scores every resume with swapped names/gender/nationality markers (rule-based agents).")
    if cfg.provider_spec != "rules" and cfg.llm_stages:
        st.warning("LLM evaluation re-runs the selected agents for 3 jobs x 11 resumes and can take a long time on CPU. "
                   "`python scripts/run_evaluation.py` runs it from the terminal.", icon=":material/hourglass_top:")
    if st.button("Run evaluation", type="primary", icon=":material/science:"):
        cfg.interview_top_k = 0  # interview questions are not scored; skip to save time
        bar = st.progress(0.0, text="Starting...")
        result = run_evaluation(cfg, name=name, fairness=fairness,
                                progress=lambda msg, frac: bar.progress(min(1.0, frac), text=msg))
        store.save_experiment(result.experiment_id, result.name, result.provider, result.config, result.metrics)
        st.session_state.last_eval = result
        bar.empty()

experiments = store.experiments()
if not experiments:
    st.info("No experiments yet. Run one above or with `python scripts/run_evaluation.py`.", icon=":material/info:")
    st.stop()

latest = experiments[0]
st.subheader(f"Latest: {latest['name']}", anchor=False)
st.caption(f"{latest['provider']} · {latest['created_at'][:19]} · id {latest['experiment_id']}")
for group, keys in METRIC_GROUPS.items():
    st.markdown(f"**{group}**")
    with st.container(horizontal=True):
        for k in keys:
            if k in latest["metrics"]:
                v = latest["metrics"][k]
                st.metric(k.replace("_", " "), f"{v:.3f}" if k != "latency_s_per_candidate" else f"{v:.2f}s",
                          help=HELP.get(k), border=True)

result = st.session_state.get("last_eval")
if result and result.experiment_id == latest["experiment_id"]:
    with st.expander("Per-job rankings vs gold grades", icon=":material/format_list_numbered:"):
        for job, info in result.per_job.items():
            st.markdown(f"**{info['title']}** · NDCG@3 {info['ndcg@3']:.3f} · JD F1 {info['jd_f1']:.3f}")
            st.dataframe(pd.DataFrame(info["ranking"]), hide_index=True,
                         column_config={"score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                                        "gold_grade": st.column_config.NumberColumn("Gold grade (0-3)")})

st.subheader("Experiment history", anchor=False)
hist = pd.DataFrame([{"experiment": e["name"], "provider": e["provider"], "created": e["created_at"][:19],
                      **e["metrics"]} for e in experiments])
st.dataframe(hist, hide_index=True)

compare_keys = ["jd_skill_f1", "resume_skill_f1", "ndcg@3", "spearman_rho", "pairwise_accuracy"]
long = hist.melt(id_vars="experiment", value_vars=[k for k in compare_keys if k in hist], var_name="metric", value_name="value")
long = long.drop_duplicates(subset=["experiment", "metric"])
st.bar_chart(long, x="metric", y="value", color="experiment", stack=False, height=320)
