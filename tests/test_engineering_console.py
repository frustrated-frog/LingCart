from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.ui.engineering_data import dataset_catalog, experiment_catalog, profile_catalog


@pytest.fixture(autouse=True)
def isolated_db():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)


def test_registry_view_model_exposes_one_slot_candidate_change():
    profiles = {item["profile_id"]: item for item in profile_catalog()}
    baseline = profiles["adaptive-v1"]
    candidate = profiles["adaptive-objective-context-v1"]
    changed = [slot for slot in candidate["mechanisms"]
               if candidate["mechanisms"].get(slot) != baseline["mechanisms"].get(slot)]
    assert changed == ["context"]
    assert candidate["research"]["status"] == "candidate"
    assert candidate["execution_sha256"]


def test_dataset_view_model_reports_real_case_and_split_counts():
    datasets = {item["dataset_id"]: item for item in dataset_catalog()}
    assert datasets["offline-core-v1"]["case_count"] == 60
    assert datasets["live-support-v1"]["splits"] == {"dev": 10, "test": 10}
    assert datasets["adaptive-agent-v1"]["turn_count"] == 21


def test_experiment_view_does_not_invent_a_comparable_report():
    experiment = experiment_catalog()[0]
    assert experiment["experiment_id"] == "objective-context-v1"
    assert experiment["status"] == "AWAITING_REPORT"
    assert experiment["report"] is None


def test_app_opens_on_agent_engineering_console():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert any("Agent 工程控制台" in item.value for item in app.markdown)
    assert any(button.label == "新建业务对话" for button in app.button)


@pytest.mark.parametrize("page", ["runs_page", "mechanisms_page", "datasets_page", "experiments_page"])
def test_engineering_pages_render_from_repository_data(page):
    source = (
        "from src.ui.styles import apply_styles\n"
        f"from src.ui.engineering_pages import {page}\n"
        "apply_styles()\n"
        f"{page}()"
    )
    app = AppTest.from_string(source).run(timeout=20)
    assert not app.exception
