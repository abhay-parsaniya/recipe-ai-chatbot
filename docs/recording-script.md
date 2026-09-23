# 2-minute demo recording — script and shot list

Two takes, one minute each: **code architecture**, then **live demo**.
Narration is written to be read at a normal pace (~150 words/minute).
Word counts are given so you can check your timing before recording.

---

## Pre-flight (do this before you hit record)

```bash
cd ~/Desktop/recipe-ai-chatbot
source .venv/bin/activate

# start both services and let them finish loading (~5s)
streamlit run app/main.py --server.headless true &
uvicorn src.api.main:app --port 8000 &

# confirm ready — must say 372213
curl -s localhost:8000/health
```

Then:

1. Open **http://localhost:8501** in a browser, send one throwaway query
   (`chicken`), and click **Start over**. This warms the index so the
   demo has no first-query pause.
2. Open the editor with exactly these four files in tabs, in this order:
   `src/data/preprocess.py`, `src/search/engine.py`,
   `src/chatbot/guardrails.py`, `evals/search_cases.json`
3. Editor font size 16+ — it will be unreadable at normal size.
4. Close Slack, mail, notifications. Hide bookmarks bar.
5. OBS: Display Capture, 1080p, 30fps, mic on a separate audio track.

---

## PART 1 — Code architecture (0:00–1:00)

### 0:00–0:12 · Show: project root / README

> "A recipe chatbot over the RecipeNLG dataset — three hundred and
> seventy-two thousand recipes, searched locally, no language model
> needed. Four layers, each swappable on its own."

### 0:12–0:28 · Show: `src/data/preprocess.py` → `build_search_text`

> "The data layer builds one searchable field per recipe. The trick is
> the weighting — extracted ingredient names three times, title twice,
> ingredient lines once. So a recipe is described by what makes it
> unusual. Saffron counts; water doesn't."

### 0:28–0:46 · Show: `src/search/engine.py` → the `search` method

> "Retrieval is TF-IDF and cosine similarity, with two corrections.
> A coverage boost, because cosine divides by length and would rank a
> three-ingredient dip above a recipe that has everything you asked
> for. And a title bonus — without it, 'banana bread' returns a banana
> sandwich, which genuinely does contain banana and bread."

### 0:46–0:54 · Show: `src/chatbot/guardrails.py`

> "Above that, the chat layer: rule-based intents, conversation memory,
> and a guardrail. Ask who the president is and it declines — instead of
> returning President Harding's Pudding Pie."

### 0:54–1:00 · Show: `evals/search_cases.json`, then terminal `pytest -q`

> "And it's measured — three hundred fifty-one tests, and a retrieval
> eval scoring precision-at-five of point eight nine five."

*(Part 1: 160 words ≈ 64s. If you run long, drop the final eval sentence — it is the least load-bearing.)*

---

## PART 2 — Live demo (1:00–2:00)

Switch to the browser. Type slowly enough to be readable; the narration
covers the wait.

### 1:00–1:18 · Type: `what can I make with chicken, tomato and onion?`

> "Asking in plain English. It strips the filler, searches on the
> ingredients, and comes back conversationally — and notice it tells me
> what each recipe needs *beyond* what I asked for, which is the thing
> that actually decides what I cook."

### 1:18–1:32 · Type: `2` … then `how do I make it?`

> "It holds context. I pick one by number, then ask how to make *it* —
> and it knows what 'it' refers to, and gives me the real method from
> the dataset."

### 1:32–1:44 · Type: `Who is the president of India?`

> "Out of domain, so it refuses cleanly rather than guessing. That
> boundary is tuned to let real cooking questions through — it's
> measured on sixty cases."

### 1:44–2:00 · Sidebar: `chicken, onion, tomato, garlic` / don't have `cheese` → **What can I make?**

> "And the main feature — I tell it what's in my kitchen. This flips the
> question: instead of how well a recipe matches my words, it ranks by
> how much of the recipe I already have, ticks off what I've got, and
> names what's missing. Cheese is excluded outright, not just ranked
> lower."

*(Part 2: 147 words — about 59 seconds, with typing covered by the narration)*

---

## If you have to cut for time

Drop the guardrail beat (1:32–1:44) — the pantry feature is the stronger
finish and needs its full 16 seconds.

## What not to claim

The LLM layer is present but switched off, and has never been run
against a live API. The sidebar says "LLM: off". Don't describe the
output as AI-generated — it's retrieval plus templates, which is a
perfectly good story on its own.
