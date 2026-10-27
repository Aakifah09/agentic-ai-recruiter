"""Headless UI tests with Streamlit's AppTest."""

import pandas  # noqa: F401  - warm heavy imports outside the AppTest script thread
import pytest
from streamlit.testing.v1 import AppTest

import app_utils
import recruiter.evaluation  # noqa: F401
import recruiter.reporting  # noqa: F401
import recruiter.storage as storage
from recruiter.llm import Environment

APP = "../streamlit_app.py"
TIMEOUT = 180  # generous: CI / loaded laptops can be slow on the first script run


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "DEFAULT_DB", tmp_path / "app.db")
    app_utils.get_store.clear()
    app_utils.get_environment.clear()
    monkeypatch.setattr(app_utils, "detect_environment", lambda: Environment("Windows", "3.11", "cpu", "none", False, []))
    monkeypatch.setattr(app_utils, "Store", lambda: storage.Store(tmp_path / "app.db"))
    yield
    app_utils.get_store.clear()


def test_analysis_flow_and_shortlist():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.session_state.provider == "rules"
    run_btn = next(b for b in at.button if b.label == "Run agent workflow")
    run_btn.click().run()
    assert not at.exception
    assert at.session_state.current_run_id

    at.switch_page("app_pages/shortlist.py").run()
    assert not at.exception
    assert any(m.label == "Candidates" and m.value == "11" for m in at.metric)


def test_review_is_saved():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    next(b for b in at.button if b.label == "Run agent workflow").click().run()
    at.switch_page("app_pages/shortlist.py").run()
    next(t for t in at.text_input if t.label == "Reviewer").input("sam")
    next(t for t in at.text_area if t.label == "Notes").input("Strong evidence for RAG")
    next(b for b in at.button if b.label == "Save review").click().run()
    assert not at.exception
    store = app_utils.get_store()
    reviews = store.reviews(at.session_state.current_run_id)
    assert next(iter(reviews.values())).notes == "Strong evidence for RAG"


@pytest.mark.parametrize("page", ["app_pages/evaluation.py", "app_pages/about.py", "app_pages/shortlist.py"])
def test_pages_render_without_data(page):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.switch_page(page).run()
    assert not at.exception
