"""Pure rendering. No searching, no LLM calls, no state mutation.

Every function here takes data and draws it. That boundary is what
keeps the UI swappable: a FastAPI + React front end would reimplement
this file and nothing else.
"""

from __future__ import annotations

import streamlit as st

from src.search import SearchResult

# Bands for the relevance badge. TF-IDF cosine scores on this corpus sit
# roughly in 0.2-0.5, so a raw percentage reads as "everything is bad".
# The wording describes what the number *is* -- word overlap -- rather
# than implying the recipe is good.
SCORE_BANDS = [
    (0.35, "strong word overlap", "#16794f"),
    (0.25, "moderate word overlap", "#8a6116"),
    (0.00, "weak word overlap", "#8a3116"),
]


def score_label(score: float) -> tuple[str, str]:
    for threshold, label, colour in SCORE_BANDS:
        if score >= threshold:
            return label, colour
    return SCORE_BANDS[-1][1], SCORE_BANDS[-1][2]


def render_relevance(result: SearchResult) -> None:
    """Show the score and say plainly what it does and does not mean."""
    label, colour = score_label(result.score)
    st.markdown(
        f"<span style='color:{colour};font-size:0.85rem'>"
        f"<b>{result.score:.0%}</b> &middot; {label}</span>",
        unsafe_allow_html=True,
    )
    if result.coverage < 1.0:
        st.caption(
            f"Has {result.coverage:.0%} of your ingredients "
            "— some are missing."
        )
    if result.matched_terms:
        st.caption("Matched: " + ", ".join(result.matched_terms[:8]))


def render_recipe_card(result: SearchResult, position: int,
                       expanded: bool = False) -> None:
    """One recipe: title, relevance, ingredients, instructions."""
    with st.expander(f"{position}. {result.title}", expanded=expanded):
        render_relevance(result)

        left, right = st.columns([1, 1.4])
        with left:
            st.markdown("**Ingredients**")
            if result.ingredients:
                for item in result.ingredients:
                    st.markdown(f"- {item}")
            else:
                st.caption("No ingredients recorded for this recipe.")

        with right:
            st.markdown("**Instructions**")
            if result.directions:
                for number, step in enumerate(result.directions, 1):
                    st.markdown(f"{number}. {step}")
            else:
                st.caption("No instructions recorded for this recipe.")

        if result.link:
            url = result.link if result.link.startswith("http") \
                else f"https://{result.link}"
            st.caption(f"Source: [{result.link}]({url})")


def render_results(results: list[SearchResult]) -> None:
    if not results:
        return
    st.caption(f"{len(results)} recipes from the database")
    for position, result in enumerate(results, 1):
        render_recipe_card(result, position, expanded=(len(results) == 1))


def render_recipes(turn: dict) -> None:
    """Draw whichever recipes this turn is about.

    A search turn carries a list; a "show me the second one" turn
    carries a single selection and an empty list. Without this the
    selected recipe would be described in prose with no card to open.
    """
    pantry = turn.get("pantry")
    if pantry:
        complete = sum(1 for m in pantry if m.is_complete)
        st.caption(f"{len(pantry)} matches · {complete} you can make right now")
        for position, match in enumerate(pantry, 1):
            render_pantry_match(match, position)
        return

    results = turn.get("results") or []
    if results:
        render_results(results)
        return
    selected = turn.get("selected")
    if selected is not None:
        render_recipe_card(selected, 1, expanded=True)


def render_pantry_match(match, position: int) -> None:
    """A pantry hit: the checklist first, the recipe behind an expander.

    The checklist is what the user came for -- "can I cook this without
    going out" -- so it is visible without a click.
    """
    ready = "ready" if match.is_complete else f"{len(match.missing)} missing"
    with st.expander(f"{position}. {match.title}  ·  {match.match:.0%}  ·  {ready}",
                     expanded=position == 1):
        left, right = st.columns(2)
        with left:
            st.markdown("**You have**")
            for item in match.have:
                st.markdown(f"- :green[✓] {item}")
        with right:
            st.markdown("**You need**")
            if match.missing:
                for item in match.missing:
                    st.markdown(f"- :red[✗] {item}")
            else:
                st.markdown("_nothing — you can make this now_")

        if match.assumed_staples:
            st.caption("Assuming you have: " + ", ".join(match.assumed_staples))

        st.divider()
        st.markdown("**Full ingredient list**")
        for item in match.ingredients:
            st.markdown(f"- {item}")
        st.markdown("**Instructions**")
        for number, step in enumerate(match.directions, 1):
            st.markdown(f"{number}. {step}")
        if match.link:
            url = match.link if match.link.startswith("http") else f"https://{match.link}"
            st.caption(f"Source: [{match.link}]({url})")


def render_pantry_panel(bot) -> None:
    """The kitchen inventory form, in the sidebar.

    A form, not a chat message: an ingredient list is structured data,
    and making the user re-type it to tweak one item would be hostile.
    """
    st.sidebar.subheader("What's in your kitchen?")
    have_text = st.sidebar.text_area(
        "I have", key="pantry_have", height=90,
        placeholder="chicken, onion, tomato, garlic",
    )
    exclude_text = st.sidebar.text_input(
        "I don't have", key="pantry_exclude", placeholder="cheese",
    )
    staples = st.sidebar.checkbox(
        "Assume I have salt, oil, water etc.", value=True, key="pantry_staples",
    )
    go = st.sidebar.button("What can I make?", use_container_width=True,
                           type="primary")
    if not go:
        return None

    from src.chatbot.pantry_query import split_items

    have = split_items(have_text)
    if not have:
        st.sidebar.warning("List at least one ingredient.")
        return None

    return {
        "have": have,
        "exclude": split_items(exclude_text),
        "assume_staples": staples,
    }


def render_turn(turn: dict) -> None:
    """Redraw one stored exchange."""
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        if turn.get("error"):
            st.error(turn["text"])
        else:
            st.write(turn["text"])
        render_recipes(turn)
        if turn.get("used_llm"):
            st.caption("Wording generated by an LLM from the recipes above.")


def render_load_failure(failure) -> None:
    st.error(failure.headline)
    if failure.fix:
        st.markdown("Run this from the project root:")
        st.code(failure.fix, language="bash")
    with st.expander("Details"):
        st.code(failure.detail)


def render_sidebar(bot) -> None:
    """Status and examples. Reads from the bot; never changes it."""
    st.sidebar.header("Recipe Assistant")
    st.sidebar.caption(
        "Recipes come from a local RecipeNLG index. Nothing is invented."
    )

    st.sidebar.subheader("Status")
    st.sidebar.write(f"Recipes indexed: **{len(bot.engine.recipes):,}**")
    if bot.llm_enabled:
        st.sidebar.write(f"LLM: **on** ({bot.settings.llm_model})")
    else:
        st.sidebar.write("LLM: **off** — plain template replies")
        st.sidebar.caption("Set ANTHROPIC_API_KEY and LLM_ENABLED=true to enable.")

    st.sidebar.subheader("Try asking")
    for example in [
        "What can I make with chicken, tomato and onion?",
        "I have shrimp, lemon and garlic",
        "show me the second one",
        "how do I make it?",
    ]:
        st.sidebar.markdown(f"- _{example}_")

    st.sidebar.subheader("Limits")
    st.sidebar.caption(
        "Matching is lexical, so \"aubergine\" won't find \"eggplant\", and "
        "\"no dairy\" can't be excluded reliably. Check the ingredients."
    )
