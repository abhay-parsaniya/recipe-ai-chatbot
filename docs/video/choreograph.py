"""Drive the REAL Chrome window through the 2-minute demo, on the clock.

Timing is deadline-based, not sleep-based: every beat targets an absolute
offset from t0, so a slow page load steals from its own beat instead of
pushing everything after it out of sync with the narration.
"""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCR = Path(sys.argv[1])
TEMPO, GAP = 1.074, 0.30
man = json.loads((SCR / "manifest.json").read_text())
DUR = [s["dur"] / TEMPO + GAP for s in man]
MARKS, acc = [], 0.0
for d in DUR:
    acc += d
    MARKS.append(acc)                      # end-time of each segment

CODE = f"file://{SCR}/code.html"
UI, API = "http://localhost:8501", "http://localhost:8000/docs"
t0 = None


def hold(i):
    """Wait until segment i is due to end."""
    target = t0 + MARKS[i]
    while time.time() < target:
        time.sleep(0.05)
    print(f"  seg{i} done at {time.time()-t0:6.2f}s (target {MARKS[i]:6.2f})",
          flush=True)


def scroll_to(pg, anchor):
    pg.evaluate("""a => document.getElementById(a)
                   .scrollIntoView({behavior:'smooth', block:'start'})""", anchor)


def ask(pg, text, delay=45):
    box = pg.locator('textarea[data-testid="stChatInputTextArea"]')
    box.click()
    box.type(text, delay=delay)
    pg.keyboard.press("Enter")


with sync_playwright() as pw:
    browser = pw.chromium.connect_over_cdp("http://localhost:9222")
    pg = browser.contexts[0].pages[0]

    if "--warm" in sys.argv:
        # Run BEFORE the recorder starts, so no cold page load and no
        # conversation reset ends up inside the captured footage.
        pg.goto(UI, wait_until="networkidle", timeout=120000)
        pg.wait_for_timeout(2500)
        try:
            pg.locator('button:has-text("Start over")').click()
            pg.wait_for_timeout(1500)
        except Exception as exc:
            print("  (no reset needed)", exc)
        pg.goto(API, wait_until="networkidle", timeout=120000)
        pg.wait_for_timeout(1500)
        pg.goto(CODE, wait_until="load")
        pg.wait_for_timeout(1200)
        print("WARM", flush=True)
        raise SystemExit(0)

    t0 = time.time()

    # ---- PART 1 : code -------------------------------------------------
    for i, anchor in enumerate(["overview", "preprocess", "engine",
                                "guardrails", "evals"]):
        scroll_to(pg, anchor)
        hold(i)

    # ---- PART 2 : the real UI -----------------------------------------
    pg.goto(UI, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_timeout(900)
    ask(pg, "what can I make with chicken, tomato and onion?")
    hold(5)

    ask(pg, "2", delay=90)
    pg.wait_for_timeout(2200)
    ask(pg, "how do I make it?")
    hold(6)

    ask(pg, "Who is the president of India?")
    hold(7)

    # Streamlit commits a text widget on blur and reruns the script.
    # Typing, then immediately clicking the button, loses the click to
    # that rerun -- so each field is committed with Tab and given time
    # to settle before the button is pressed.
    have = pg.locator('[data-testid="stSidebar"] textarea')
    have.click()
    have.type("chicken, onion, tomato, garlic", delay=42)
    pg.keyboard.press("Tab")
    pg.wait_for_timeout(900)

    avoid = pg.locator('[data-testid="stSidebar"] input[type="text"]')
    avoid.click()
    avoid.type("cheese", delay=55)
    pg.keyboard.press("Tab")
    pg.wait_for_timeout(900)

    pg.locator('button:has-text("What can I make?")').click()
    pg.wait_for_timeout(2500)
    try:                       # bring the new cards into view
        pg.evaluate("window.scrollTo({top:0, behavior:'smooth'})")
    except Exception as exc:
        print("  (scroll skipped)", exc)
    hold(8)

    pg.goto(API, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_timeout(1400)
    try:
        pg.locator('span:has-text("/pantry")').first.click()
        pg.wait_for_timeout(700)
        pg.evaluate("window.scrollBy({top:260, behavior:'smooth'})")
    except Exception as e:
        print("  (swagger expand skipped)", e)
    hold(9)

    print(f"\nrun finished in {time.time()-t0:.2f}s (target {MARKS[-1]:.2f}s)")
