"""Shared Streamlit helpers: cached resources and settings -> workflow config."""

from __future__ import annotations

import streamlit as st

from recruiter.agents.matcher import ScoringConfig
from recruiter.llm import Environment, detect_environment
from recruiter.orchestrator import WorkflowConfig
from recruiter.storage import Store

STAGE_LABELS = {
    "jd": "Job analysis",
    "resume": "Resume extraction",
    "narrative": "Match narrative",
    "interview": "Interview questions",
}


@st.cache_resource
def get_store() -> Store:
    return Store()


@st.cache_data(ttl=60, show_spinner=False)
def get_environment() -> Environment:
    return detect_environment()


def workflow_config() -> WorkflowConfig:
    s = st.session_state
    return WorkflowConfig(
        provider_spec=s.provider,
        temperature=s.temperature,
        blind_review=s.blind_review,
        interview_top_k=s.interview_top_k,
        questions_per_candidate=s.questions_per_candidate,
        llm_stages=tuple(s.llm_stages or ()),
        scoring=ScoringConfig(**{k: s[f"sc_{k}"] for k in ScoringConfig.model_fields}),
    )


def score_color(score: float) -> str:
    return "green" if score >= 70 else "orange" if score >= 45 else "gray"
