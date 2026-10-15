#!/usr/bin/env python3
"""Run a safe configuration or live Model Gateway smoke check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from src.errors import classify_exception  # noqa: E402
from src.llm import extract_usage, get_llm, get_model_status  # noqa: E402


@tool
def connection_echo(text: str) -> str:
    """回显模型提供的文本，用于验证工具调用协议。"""

    return text


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def run_check(*, purpose: str, check_tools: bool, config_only: bool) -> tuple[dict, int]:
    status = get_model_status()
    if config_only or not status["configured"]:
        return status, 0 if status["configured"] else 1

    try:
        model = get_llm(purpose=purpose)
        if check_tools:
            active_model = model.bind_tools([connection_echo])
            response = active_model.invoke([
                HumanMessage(content="请调用 connection_echo，参数 text 必须是“模型接入正常”。不要直接回答。")
            ])
            tool_calls = getattr(response, "tool_calls", [])
            if not tool_calls:
                return {
                    "status": "failed",
                    "message": "模型文本调用成功，但没有按要求返回工具调用",
                    "error_code": "MODEL_TOOL_CALL_CHECK_FAILED",
                    "gateway": get_model_status(),
                }, 2
            call = tool_calls[0]
            tool_result = connection_echo.invoke(call.get("args") or {})
            check_result = {"tool": call.get("name"), "result": tool_result}
        else:
            active_model = model
            response = active_model.invoke([HumanMessage(content="请只回复：模型接入正常")])
            check_result = {"content": _content_text(response.content)[:500]}
        return {
            "status": "ok",
            "message": "模型实时自检通过",
            "purpose": purpose,
            "provider": active_model.last_call,
            "usage": extract_usage(response),
            "result": check_result,
        }, 0
    except Exception as exc:
        error = classify_exception(exc, source="model")
        return {
            "status": "failed",
            "message": "模型实时自检失败",
            "error_code": error.code,
            "retryable": error.retryable,
            "gateway": get_model_status(),
        }, 1


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 Model Gateway 配置和实时模型调用")
    parser.add_argument(
        "--purpose",
        choices=("default", "router", "goal", "planner", "agent", "final_response"),
        default="default",
        help="选择要验证的模型用途",
    )
    parser.add_argument("--check-tools", action="store_true", help="同时验证模型是否能返回工具调用")
    parser.add_argument("--config-only", action="store_true", help="只检查配置，不向 Provider 发请求")
    args = parser.parse_args()
    result, exit_code = run_check(
        purpose=args.purpose,
        check_tools=args.check_tools,
        config_only=args.config_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
