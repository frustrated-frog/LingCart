from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.workspace import list_runs


@pytest.fixture(autouse=True)
def isolated_db():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)


def test_workspace_order_query_creates_an_inspectable_record():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=15)
    assert not app.exception
    app.button(key="quick_查物流").click().run(timeout=15)
    assert not app.exception
    assert "运输中" in app.session_state["messages"][-1]["content"]
    assert list_runs()[0]["result"]["trace"][1]["tool"] == "track_shipment"


def test_workspace_shopping_scenario_returns_actual_catalog_results():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=15)
    app.button(key="scenario_0").click().run(timeout=15)
    assert not app.exception
    result = app.session_state["messages"][-1]["result"]
    assert result["agent"] == "shopping"
    assert result["trace"][0]["result"]["products"]


def test_catalog_selection_builds_comparison_from_two_real_records():
    app = AppTest.from_string("from src.ui.pages import init_state, products_page\ninit_state()\nproducts_page()").run(timeout=15)
    app.checkbox(key="compare_P1309S").check().run(timeout=15)
    app.checkbox(key="compare_P1301S").check().run(timeout=15)
    assert not app.exception
    assert app.session_state["compare_ids"] == ["P1309S", "P1301S"]
    assert any(button.label == "让助手对比这些商品" for button in app.button)
    app.button(key="clear_comparison").click().run(timeout=15)
    assert app.session_state["compare_ids"] == []
    assert not app.checkbox(key="compare_P1309S").value


def test_order_page_request_refreshes_pending_action_in_sidebar():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=15)
    app.session_state["queued_prompt"] = "帮我支付订单 1003"
    app.run(timeout=15)
    assert not app.exception
    assert app.session_state["memory"].pending_action["order_id"] == 1003
    assert next(button for button in app.button if button.label == "新建对话").disabled
    assert tools.get_payment_status(1003)["status"] == "pending"
