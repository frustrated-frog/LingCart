"""Capture real screenshots from the running React workspace.

Usage: python scripts/capture_screenshots.py
(requires the production build/API running at http://localhost:8000,
 and `pip install playwright && playwright install chromium`)
"""

import time

from playwright.sync_api import sync_playwright

URL = "http://localhost:8000/sandbox"
OUT_DIR = "demo"


def send_message(page, text: str) -> None:
    textarea = page.get_by_label("消费需求")
    textarea.click()
    textarea.fill(text)
    page.get_by_label("发送需求").click()


def wait_for_reply(page, timeout_s: int = 30) -> None:
    page.wait_for_selector("text=正在执行并核验结果", state="visible", timeout=timeout_s * 1000)
    page.wait_for_selector("text=正在执行并核验结果", state="detached", timeout=timeout_s * 1000)
    time.sleep(0.5)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(URL)
        page.wait_for_selector("textarea", timeout=20000)
        time.sleep(1)

        # 1. Order lookup
        send_message(page, "Hi, where is my order 1004?")
        wait_for_reply(page)
        page.screenshot(path=f"{OUT_DIR}/01_order_lookup.png")

        # 2. Guardrail refusal (order 1002 is shipped, not delivered)
        send_message(page, "I'd like a refund for order 1002, it's the wrong item")
        wait_for_reply(page)
        page.screenshot(path=f"{OUT_DIR}/02_guardrail_refusal.png")

        # Open the run inspector to capture the durable execution spine.
        page.get_by_text("运行检查", exact=True).click()
        page.wait_for_selector("text=Execution Spine", timeout=20000)
        page.screenshot(path=f"{OUT_DIR}/02b_guardrail_trace_expanded.png")

        # 3. FAQ + engineering console telemetry
        page.goto(URL)
        page.wait_for_selector("textarea", timeout=20000)
        send_message(page, "What is your return policy?")
        wait_for_reply(page)
        page.get_by_text("Agent 控制台", exact=True).click()
        page.wait_for_selector("text=Harness Profiles", timeout=20000)
        time.sleep(0.5)
        page.screenshot(path=f"{OUT_DIR}/03_faq_and_sidebar.png")

        browser.close()
        print("Saved screenshots to demo/")


if __name__ == "__main__":
    main()
