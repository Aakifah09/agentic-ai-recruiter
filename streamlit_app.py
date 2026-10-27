"""Agentic AI Recruiter - Streamlit entry point.

Run with:  streamlit run streamlit_app.py
"""

import streamlit as st

from app_utils import STAGE_LABELS, get_environment
from recruiter.agents.matcher import ScoringConfig

st.set_page_config(page_title="Agentic AI Recruiter", page_icon=":material/groups:", layout="wide")

env = get_environment()

# --------------------------------------------------------------------------- session state
defaults = {
    "provider": env.recommended_provider(),
    "temperature": 0.1,
    "blind_review": True,
    "interview_top_k": 3,
    "questions_per_candidate": 6,
    "llm_stages": ["jd", "interview"],
    "current_run_id": None,
    **{f"sc_{k}": v for k, v in ScoringConfig().model_dump().items()},
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)



def reset_scoring() -> None:
    for k, v in ScoringConfig().model_dump().items():
        st.session_state[f"sc_{k}"] = v


# --------------------------------------------------------------------------- sidebar settings
with st.sidebar:
    st.subheader(":material/tune: Settings")
    options = env.provider_options()
    if st.session_state.provider not in options:
        st.session_state.provider = "rules"
    st.selectbox(
        "Model provider", options, key="provider",
        format_func=lambda o: "Rule-based (offline, instant)" if o == "rules" else o,
        help="Local Ollama models are detected automatically. External APIs appear when OPENAI_API_KEY or ANTHROPIC_API_KEY is set.",
    )
    if st.session_state.provider != "rules":
        st.pills("Agents that use the LLM", list(STAGE_LABELS), selection_mode="multi", key="llm_stages",
                 format_func=STAGE_LABELS.get,
                 help="On CPU-only machines each LLM call takes roughly 1-2 minutes with a 3B model. "
                      "Unselected agents use deterministic rules.")
        st.slider("Temperature", 0.0, 1.0, key="temperature", step=0.05)
    st.toggle("Blind review (hide names)", key="blind_review",
              help="Show candidates as 'Candidate A, B...'. Names and contact details are always redacted before analysis.")
    st.number_input("Interview plans for top-k", 0, 20, key="interview_top_k")
    st.number_input("Questions per candidate", 3, 12, key="questions_per_candidate")

    with st.expander("Scoring policy"):
        st.slider("Skills weight", 0.0, 1.0, step=0.05, key="sc_skill_weight")
        st.slider("Experience weight", 0.0, 1.0, step=0.05, key="sc_experience_weight")
        st.slider("Qualifications weight", 0.0, 1.0, step=0.05, key="sc_qualification_weight")
        st.slider("Credit for listed-only skills", 0.0, 1.0, step=0.05, key="sc_listed_credit",
                  help="Skills that appear only in a skills list, without evidence of use.")
        st.slider("Credit for related skills", 0.0, 1.0, step=0.05, key="sc_related_credit",
                  help="E.g. Azure experience when AWS is requested.")
        st.button("Reset policy", icon=":material/restart_alt:", on_click=reset_scoring)

    with st.expander("Environment"):
        st.caption(f"{env.os} · Python {env.python}")
        st.caption(f"CPU: {env.cpu}")
        st.caption(f"GPU: {env.gpu}")
        st.caption("Ollama: " + (f"running, {len(env.ollama_models)} model(s)" if env.ollama_running else "not detected"))

# --------------------------------------------------------------------------- navigation
page = st.navigation(
    [
        st.Page("app_pages/analyze.py", title="New analysis", icon=":material/person_search:", default=True),
        st.Page("app_pages/shortlist.py", title="Shortlist & review", icon=":material/fact_check:"),
        st.Page("app_pages/evaluation.py", title="Evaluation", icon=":material/monitoring:"),
        st.Page("app_pages/about.py", title="How it works", icon=":material/account_tree:"),
    ],
    position="top",
)
page.run()
