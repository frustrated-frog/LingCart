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


def test_preferences_page_saves_and_forgets_confirmed_budget():
    app = AppTest.from_string('from src.ui.pages import init_state, preferences_page\ninit_state()\npreferences_page()').run(timeout=15)
    next(b for b in app.button if b.label == '创建我的偏好档案').click().run(timeout=15)
    app.text_input[0].set_value('300')
    app.checkbox[0].check()
    next(b for b in app.button if b.label == '保存偏好').click().run(timeout=15)
    assert not app.exception
    from src.preferences import PreferenceStore
    profile = app.session_state['profile_id']
    assert PreferenceStore().list(profile)[0]['value']['value'] == 300
    app.button(key='forget_max_price').click().run(timeout=15)
    assert not app.exception
    assert PreferenceStore().list(profile) == []


def test_active_run_panel_accepts_steering_command():
    from src.session_control import ControlStore
    claim = ControlStore().claim('active-ui', 'one', '推荐耳机', {})
    app = AppTest.from_string('from src.ui.pages import init_state, home_page\ninit_state()\nhome_page()')
    app.session_state['conversation_id'] = 'active-ui'
    app.session_state['active_run_id'] = claim['run_id']
    app.run(timeout=15)
    assert not app.exception
    app.text_input[0].set_value('预算改成300元')
    next(b for b in app.button if b.label == '发送补充要求').click().run(timeout=15)
    assert not app.exception
    commands = ControlStore().commands('active-ui', claim['run_id'])
    assert commands[0]['kind'] == 'STEER'
    assert commands[0]['payload']['revision'] == 1


def test_active_run_form_does_not_silently_upgrade_a_stale_revision():
    from src.session_control import ControlStore
    store = ControlStore()
    claim = store.claim('stale-ui', 'one', '推荐耳机', {})
    app = AppTest.from_string('from src.ui.pages import init_state, home_page\ninit_state()\nhome_page()')
    app.session_state['conversation_id'] = 'stale-ui'
    app.session_state['active_run_id'] = claim['run_id']
    app.run(timeout=15)
    store.enqueue('stale-ui', claim['run_id'], 'STEER', message='预算500', command_id='other-window', expected_revision=0)
    app.text_input[0].set_value('预算300')
    next(b for b in app.button if b.label == '发送补充要求').click().run(timeout=15)
    assert not app.exception
    assert len(store.commands('stale-ui', claim['run_id'])) == 1
    assert any('版本已变化' in e.value for e in app.error)
