"""Screenshots of the running app for the report and deck (Playwright, 1440x900).

    streamlit run app.py --server.headless true --server.port 8501 &
    python scripts/make_screenshots_final.py

Writes docs/screenshots/final/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots" / "final"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8501"
SPINNER = '[data-testid="stSpinner"]'
PANEL = '[data-testid="stTabPanel"]:visible'


def settle(page, timeout=900_000):
    page.wait_for_timeout(1200)
    page.wait_for_function(f"document.querySelectorAll('{SPINNER}').length === 0", timeout=timeout)
    page.wait_for_timeout(2000)


def fresh(page, query=""):
    page.goto(BASE + query)
    page.wait_for_selector('[data-testid="stTab"]', timeout=120_000)
    settle(page)


def tab(page, name):
    page.get_by_role("tab", name=name, exact=True).click()
    settle(page)


def ask(page, text):
    box = page.locator(PANEL).first.get_by_label("Ask something")
    box.fill(text)
    page.locator(PANEL).first.get_by_role("button", name="Send").click()
    settle(page)


def build(page, label):
    page.locator(PANEL).first.get_by_text(label, exact=True).click()
    settle(page)


def shot(page, name, full=False):
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
    print("  ", name)


def main(only=None):
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1440, "height": 1500}, device_scale_factor=1.5)

        def want(n):
            return not only or n in only

        if want("01"):
            fresh(page)
            build(page, "Model A: routing only")
            ask(page, "Write a Python function merge_sorted(a, b) that merges two sorted lists into one sorted list. "
                      "Your code should pass this test: assert merge_sorted([1, 4], [2, 3]) == [1, 2, 3, 4]")
            shot(page, "01_route_coding", full=True)
        if want("01b"):
            fresh(page)
            build(page, "Model A: routing only")
            ask(page, "A shirt costs 800 rupees after a 20% discount. What was the original price? "
                      "End with 'Answer: <number>'.")
            shot(page, "01b_local_first_maths", full=True)
        if want("02"):
            fresh(page)
            build(page, "Model A: routing only")
            ask(page, "Who wrote the novel Midnight's Children?")
            shot(page, "02_route_factual", full=True)
        if want("03"):
            fresh(page, "?wa=0.3&wc=0.6&wl=0.1")
            build(page, "Model A: routing only")
            ask(page, "Who wrote the novel Midnight's Children?")
            shot(page, "03_route_factual_cost_weight", full=True)
        if want("04"):
            fresh(page)
            page.locator(PANEL).first.get_by_role("button", name="Wipe memory").click()
            settle(page)
            ask(page, "My name is Arjun and I work at PayLoop as a data engineer.")
            page.locator(PANEL).first.get_by_role("button", name="New session").click()
            settle(page)
            ask(page, "Where do I work?")
            shot(page, "04_memory_private_local", full=True)
        if want("05"):
            fresh(page)
            build(page, "Model C: RAG only")
            ask(page, "Who chairs the Helix board?")
            shot(page, "05_rag_passages", full=True)
        if want("06"):
            fresh(page)
            tab(page, "Compare A/B/C/D")
            page.locator(PANEL).first.get_by_role("button", name="Run through baseline, A, B, C, D").click()
            settle(page)
            shot(page, "06_compare_builds", full=True)
        for n, t in [("07", "Scorecards"), ("08", "Results"), ("09", "Privacy & cost")]:
            if want(n):
                fresh(page)
                tab(page, t)
                shot(page, f"{n}_{t.split()[0].lower()}", full=True)
        b.close()


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
