# Recipe Recommendation Chatbot

A domain-specific chatbot that recommends recipes from the
[`mbien/recipe_nlg`](https://huggingface.co/datasets/mbien/recipe_nlg) dataset.

Given ingredients or a free-text request ("something with chicken and rice, no dairy"),
the system returns the closest matching recipes with their ingredients and directions.

> **Status:** project scaffolding only. No chatbot logic implemented yet.

## Architecture

```
user -> Streamlit UI -> FastAPI -> retrieval engine (TF-IDF + cosine similarity) -> recipe data
```

The retrieval layer is deliberately classical (scikit-learn) so the project runs
locally with no GPU. An LLM can be layered on later to rewrite the answer in
natural language, without changing the retrieval core.

## Project structure

```
recipe-ai-chatbot/
├── app/                   # Streamlit front-end
│   ├── main.py            # Page flow only
│   ├── components.py      # Pure rendering
│   └── state.py           # Session + cached index load
├── data/
│   ├── raw/               # Unmodified dataset dumps
│   ├── processed/         # Cleaned, model-ready data
│   └── artifacts/         # Fitted vectorizers, matrices
├── notebooks/             # Exploration and prototyping
├── src/
│   ├── api/               # FastAPI app, routes, schemas, sessions
│   ├── chatbot/           # Guardrails, intents, query parsing, history
│   ├── config/            # Settings loaded from environment
│   ├── data/              # Loading, cleaning, preprocessing
│   ├── llm/               # Optional response generation
│   │   └── providers/     # Vendor SDKs live here and nowhere else
│   ├── search/            # Vectorization and similarity search
│   └── utils/             # Shared helpers (logging, text cleanup)
├── scripts/               # Runnable entry points (explore, build dataset)
├── tests/                 # Pytest suite
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv          # needs python3-venv on Debian/Ubuntu
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Getting the dataset

`mbien/recipe_nlg` is a **manual download** dataset: the Hugging Face loader
cannot fetch it for you. Download RecipeNLG from
<https://recipenlg.cs.put.poznan.pl/> (accept the licence), unzip it, and put
`full_dataset.csv` in `data/raw/`.

```bash
python -m scripts.explore_dataset --limit 5000   # columns, size, samples
python -m scripts.build_dataset --limit 50000    # -> data/processed/recipes.parquet
python -m scripts.build_index                    # -> data/artifacts/tfidf_index.joblib
python -m scripts.chat                           # talk to the bot
```

## Searching

```python
from src.search import RecipeSearch

engine = RecipeSearch.load()                 # or .from_processed() to refit
for hit in engine.search("chicken rice garlic", top_k=5):
    print(hit.score, hit.title, hit.matched_terms)
```

## Chatting

```python
from src.chatbot import RecipeChatbot, Conversation

bot = RecipeChatbot.from_index()
chat = Conversation()                 # one per user; the engine is shared
print(bot.respond("what can I make with chicken and rice?", chat).text)
print(bot.respond("show me the second one", chat).text)
print(bot.respond("how do I make it?", chat).text)
```

### Ingredient-based recommendation

The headline feature. Search asks *"how much of my query is in this
recipe"*; a pantry asks the opposite — *"how much of this recipe do I
already have"*. A recipe matching all three words you typed is useless if
it needs nine other things you don't own.

```
you > I have chicken, onion, tomato and garlic but no cheese

bot > Central American Rice   —   80% match
         [x] onion   [x] tomato   [x] garlic   [x] chicken
         [ ] rice   (missing)
         assuming you have: oil, water
```

The **missing list is the point** — "80%" doesn't tell you whether to go
to the shop. Excluded ingredients remove a recipe entirely rather than
just demoting it. Staples (salt, oil, water…) are assumed by default and
listed so the assumption is visible; toggle with `assume_staples`.

Available three ways: type an inventory in chat, use the sidebar form in
Streamlit, or `POST /pantry`. Chat only switches to pantry mode on a real
inventory — an exclusion plus two ingredients, "only", or "I have" with
three or more items. `"what can I make with chicken and tomato"` stays an
ordinary search.

Matching is approximate and says so: `"chicken"` matches
`"chicken breasts"`, which also means `"tomato"` matches
`"tomato sauce"`. The checklist keeps that visible.

### Domain boundary

A rule-based guardrail (`src/chatbot/guardrails.py`) runs before retrieval
and refuses anything that isn't cooking:

```
you > Who is the president of India?
bot > I'm a recipe assistant. I can help you find recipes, ingredients,
      cooking instructions, and meal ideas.
```

Without it the bot searched the corpus for "who president india" and
returned *President Harding's Pudding Pie* as an answer.

| Allowed | Refused |
|---|---|
| Recipes, ingredients, cooking | Politics, weather, general knowledge |
| Meal ideas, substitutions | Coding, medical advice, entertainment |

The filter is tuned for **precision over recall**. A refused cooking
question is a visible failure; a stray question that slips through just
gets "I couldn't find anything matching that", which is already fine. So
any message with a real food signal is allowed even when it also mentions
an off-limits topic -- *"what can I cook that's good for a cold"* is a
recipe request, not a medical one. Refusals never reach the LLM: nothing
was retrieved, so there would be nothing to ground against.

Intent detection is rule-based (`src/chatbot/intents.py`) -- no LLM. The
layers are split so each can be replaced alone: `intents` (understanding),
`query` (message -> search terms), `responses` (wording), `history`
(transcript + what "it" refers to), `bot` (routing only).

## Optional LLM layer

Off by default -- everything above works with no API key. When enabled, the
LLM **phrases** the reply; it never decides what the recipes are:

```
user query -> TF-IDF search -> top K recipes -> LLM (given only those) -> reply
```

```bash
export ANTHROPIC_API_KEY=...       # never hardcoded, never committed
echo "LLM_ENABLED=true" >> .env
python -m scripts.chat
python -m scripts.show_prompt "chicken tomato onion"   # inspect, send nothing
```

Three guarantees, each covered by tests:

- **Retrieval and generation stay separate.** `ResponseGenerator` takes recipes
  as an argument; it holds no engine and cannot search.
- **It never raises.** Missing key, rate limit, network error, bad model name,
  empty reply -- all fall back to the template text. The user still gets their
  recipes, just plainer.
- **Output is checked, not trusted.** A reply quoting a recipe title that was
  never retrieved is discarded. Prompting reduces invention; it does not prove
  absence, so there is a mechanical check too.

Swapping provider means adding one module under `src/llm/providers/` and one
line in `build_provider`. Nothing outside that directory imports a vendor SDK.

Retrieval is TF-IDF + cosine similarity over `search_text` (NER x3, title
x2, ingredients x1), then a **coverage boost**:

```
score = cosine_similarity x coverage ** COVERAGE_WEIGHT
```

`coverage` is the fraction of the query's known words a recipe contains.
Cosine divides by vector length, so without this a three-ingredient
guacamole matching two of your three words outranks an eleven-ingredient
cacciatore matching all three. Only unigrams count -- "onion, tomato" and
"tomato, onion" are the same ingredients. Set `COVERAGE_WEIGHT=0` to get
plain cosine back. Each result carries `similarity` and `coverage`
separately, so the ranking is always inspectable.

It is still lexical, not semantic: "aubergine" will not find "eggplant",
and negations like "no dairy" match recipes that *contain* dairy. Those
gaps are what an embedding or LLM layer would close later.

## Running the app

```bash
streamlit run app/main.py
```

The UI is a thin layer: `app/main.py` handles page flow, `app/components.py`
draws, `app/state.py` caches the index for the server process. None of them
search, rank or prompt -- they call `RecipeChatbot.respond()` and render the
reply. Recipe cards show the title, ingredients, instructions, source link and
a relevance badge that says *word overlap*, not quality.

## API

```bash
uvicorn src.api.main:app --reload     # docs at http://localhost:8000/docs
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Status, recipe count, whether the LLM is on |
| `POST` | `/search` | Ranked recipes. No conversation, no LLM |
| `POST` | `/pantry` | What you can cook from an ingredient list |
| `POST` | `/chat` | One conversational turn; returns a `session_id` |
| `GET` | `/chat/{id}/history` | Full transcript + current selection |
| `DELETE` | `/chat/{id}` | End a session |
| `GET` | `/recipes/{id}` | One recipe by id |

```bash
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"message":"what can I make with chicken and rice?"}'
# -> {"session_id":"...", "text":"...", "results":[...], "used_llm":false}
```

Send `session_id` back on each turn to keep context. Sessions live in
process memory, bounded and TTL'd; clients cannot choose their own id.
A missing index gives `503` with the command to fix it, while `/health`
still answers `200 degraded` -- restarting a container will not conjure
an index.

**Before deploying:** there is no authentication and CORS is wide open.
Both are fine for a local prototype and wrong for anything public.

## Tests

```bash
pytest
```

## Roadmap

1. ~~Data ingestion and cleaning from RecipeNLG~~ done
2. ~~TF-IDF index + cosine-similarity retrieval~~ done
3. ~~Rule-based chat layer with conversation history~~ done
4. ~~Grounded LLM response generation (optional)~~ done
5. ~~Streamlit chat interface~~ done
6. ~~FastAPI service~~ done
