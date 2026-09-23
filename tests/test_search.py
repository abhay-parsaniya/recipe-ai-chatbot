import pandas as pd
import pytest

from src.config.settings import Settings
from src.data.preprocess import preprocess
from src.search import RecipeSearch, SearchResult


@pytest.fixture
def engine(tmp_path) -> RecipeSearch:
    raw = pd.DataFrame(
        {
            "title": [
                "Garlic Chicken Rice",
                "Chocolate Banana Bread",
                "Beef Chili",
                "Lemon Garlic Shrimp",
                "Vanilla Pancakes",
            ],
            "ingredients": [
                '["2 chicken breasts", "1 c. rice", "3 cloves garlic"]',
                '["3 bananas", "2 c. flour", "1/2 c. cocoa"]',
                '["1 lb. ground beef", "2 cans kidney beans", "chili powder"]',
                '["1 lb. shrimp", "4 cloves garlic", "1 lemon"]',
                '["2 c. flour", "1 tsp. vanilla", "2 eggs", "milk"]',
            ],
            "directions": [
                '["Cook the rice.", "Fry the chicken."]',
                '["Mash bananas.", "Bake 60 minutes."]',
                '["Brown the beef.", "Simmer with beans."]',
                '["Saute garlic.", "Add shrimp."]',
                '["Whisk batter.", "Fry on a griddle."]',
            ],
            "NER": [
                '["chicken", "rice", "garlic"]',
                '["bananas", "flour", "cocoa"]',
                '["ground beef", "kidney beans", "chili powder"]',
                '["shrimp", "garlic", "lemon"]',
                '["flour", "vanilla", "eggs", "milk"]',
            ],
            "link": ["a.com", "b.com", "c.com", "d.com", "e.com"],
            "source": ["Gathered"] * 5,
        }
    )
    # min_df=1: the fixture is tiny, the production default of 2 would
    # discard almost the entire vocabulary.
    settings = Settings(
        tfidf_min_df=1,
        tfidf_max_df=1.0,
        artifacts_dir=tmp_path,
        processed_data_dir=tmp_path,
    )
    return RecipeSearch(preprocess(raw), settings).fit()


def test_fit_builds_matrix_with_one_row_per_recipe(engine):
    assert engine.is_fitted
    assert engine.matrix.shape[0] == 5
    assert engine.matrix.shape[1] > 0


def test_search_ranks_the_obvious_match_first(engine):
    results = engine.search("chicken and rice with garlic", top_k=3)
    assert results[0].title == "Garlic Chicken Rice"
    assert isinstance(results[0], SearchResult)


def test_scores_are_descending_and_in_unit_range(engine):
    results = engine.search("garlic", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 for s in scores)


def test_top_k_limits_the_result_count(engine):
    assert len(engine.search("flour", top_k=1)) == 1
    assert len(engine.search("garlic", top_k=2)) <= 2


def test_unknown_query_returns_nothing_rather_than_noise(engine):
    assert engine.search("xylophone tungsten") == []


def test_min_score_filters_weak_matches(engine):
    assert engine.search("garlic", top_k=5, min_score=0.99) == []


def test_results_carry_the_data_the_chatbot_needs(engine):
    result = engine.search("shrimp lemon", top_k=1)[0]
    assert result.title == "Lemon Garlic Shrimp"
    assert "1 lb. shrimp" in result.ingredients
    assert result.directions
    assert result.link == "d.com"
    assert "shrimp" in result.matched_terms


def test_result_is_json_serializable(engine):
    import json

    payload = engine.search("beef", top_k=1)[0].to_dict()
    assert json.loads(json.dumps(payload))["title"] == "Beef Chili"


def test_query_uses_corpus_vocabulary_not_its_own(engine):
    """transform, not fit_transform: query space == recipe space."""
    vector = engine.vectorize_query("garlic")
    assert vector.shape[1] == engine.matrix.shape[1]


def test_save_and_load_roundtrip(engine, tmp_path):
    engine.recipes.to_parquet(engine.settings.processed_recipes_path, index=False)
    path = engine.save()
    assert path.exists()

    reloaded = RecipeSearch.load(path, engine.settings)
    before = engine.search("garlic chicken", top_k=3)
    after = reloaded.search("garlic chicken", top_k=3)
    assert [r.title for r in before] == [r.title for r in after]
    assert [r.score for r in before] == [r.score for r in after]


def test_unfitted_engine_refuses_to_search():
    df = pd.DataFrame({"search_text": ["chicken rice"], "title": ["x"]})
    with pytest.raises(RuntimeError):
        RecipeSearch(df).search("chicken")


def test_missing_search_text_column_is_rejected():
    with pytest.raises(KeyError):
        RecipeSearch(pd.DataFrame({"title": ["x"]}))


# -- coverage boost -----------------------------------------------------

FILLERS = ["rice", "stock", "butter", "flour", "cheese", "cream", "parsley",
           "paprika", "celery", "carrot", "thyme", "bay leaf", "pepper",
           "oil", "sugar"]


@pytest.fixture
def coverage_engine(tmp_path) -> RecipeSearch:
    """A short 2-of-3 match vs a long 3-of-3 match.

    This is exactly the case plain cosine gets wrong: the short recipe
    puts a larger share of its length on the matched words, so it wins
    despite missing an ingredient the user asked for. Neither title
    contains a query word, so the comparison is about the ingredient
    lists alone.
    """
    long_items = ", ".join(f'"{f}"' for f in FILLERS)
    raw = pd.DataFrame({
        "title": ["Fresh Salsa", "Sunday Casserole"],
        "ingredients": [
            '["2 tomatoes", "1 onion"]',
            f'["1 chicken", "2 tomatoes", "1 onion", {long_items}]',
        ],
        "directions": ['["Chop."]', '["Brown.", "Bake."]'],
        "NER": [
            '["tomatoes", "onion"]',
            f'["chicken", "tomatoes", "onion", {long_items}]',
        ],
        "link": ["a.com", "b.com"],
        "source": ["Gathered"] * 2,
    })
    settings = Settings(tfidf_min_df=1, tfidf_max_df=1.0,
                        artifacts_dir=tmp_path, processed_data_dir=tmp_path)
    return RecipeSearch(preprocess(raw), settings).fit()


def test_plain_cosine_prefers_the_short_partial_match(coverage_engine):
    """Documents the defect the boost exists to fix."""
    coverage_engine.settings.coverage_weight = 0.0
    results = coverage_engine.search("chicken tomato onion", top_k=2)
    assert results[0].title == "Fresh Salsa"        # has no chicken


def test_coverage_boost_promotes_the_full_match(coverage_engine):
    coverage_engine.settings.coverage_weight = 1.0
    results = coverage_engine.search("chicken tomato onion", top_k=2)
    assert results[0].title == "Sunday Casserole"


def test_coverage_counts_only_known_query_words(coverage_engine):
    """"tomato" is not in this corpus ("tomatoes" is), so it is not counted.

    An unmatchable term would penalise every recipe equally.
    """
    results = {r.title: r for r in
               coverage_engine.search("chicken tomato onion", top_k=2)}
    assert results["Sunday Casserole"].coverage == 1.0    # chicken + onion
    assert results["Fresh Salsa"].coverage == 0.5         # onion only


def test_score_is_similarity_times_coverage(coverage_engine):
    for result in coverage_engine.search("chicken tomato onion", top_k=2):
        assert result.score == pytest.approx(
            result.similarity * result.coverage, abs=1e-3)


def test_partial_matches_are_demoted_not_removed(coverage_engine):
    results = coverage_engine.search("chicken tomato onion", top_k=5)
    assert {r.title for r in results} == {"Fresh Salsa", "Sunday Casserole"}


def test_weight_zero_restores_plain_cosine(coverage_engine):
    coverage_engine.settings.coverage_weight = 0.0
    for result in coverage_engine.search("chicken tomato onion", top_k=2):
        assert result.score == pytest.approx(result.similarity)
        assert result.coverage == 1.0


def test_single_word_query_is_unaffected(coverage_engine):
    """One term means coverage is 1.0 for anything that matches at all."""
    for result in coverage_engine.search("chicken", top_k=2):
        assert result.coverage == 1.0
        assert result.score == pytest.approx(result.similarity)


def test_bigrams_do_not_count_toward_coverage(coverage_engine):
    """Word order is an accident: "onion chicken" must score like "chicken onion"."""
    forward = {r.title: r.coverage
               for r in coverage_engine.search("chicken onion", top_k=2)}
    reverse = {r.title: r.coverage
               for r in coverage_engine.search("onion chicken", top_k=2)}
    assert forward == reverse
    assert forward["Sunday Casserole"] == 1.0


def test_min_score_applies_to_the_boosted_score(coverage_engine):
    """A recipe above the threshold on cosine can fall below it once
    coverage is applied -- the filter must see the final score."""
    weak = min(coverage_engine.search("chicken tomato onion", top_k=5),
               key=lambda r: r.score)
    assert weak.similarity > weak.score          # the boost pushed it down

    survivors = coverage_engine.search("chicken tomato onion", top_k=5,
                                       min_score=weak.similarity)
    assert weak.title not in {r.title for r in survivors}


# -- result diversity ---------------------------------------------------

@pytest.fixture
def repeated_title_engine(tmp_path) -> RecipeSearch:
    """Five different recipes that share a name.

    Not duplicate data -- the real corpus holds 100 distinct recipes
    called "Banana Bread". Without a cap, a search for one fills the
    page with what looks like the same answer five times.
    """
    raw = pd.DataFrame({
        "title": ["Banana Bread"] * 5 + ["Banana Muffins"],
        "ingredients": [
            f'["{n} bananas", "{n} c. flour", "sugar"]' for n in range(1, 6)
        ] + ['["2 bananas", "1 c. oats"]'],
        "directions": ['["Mash.", "Bake."]'] * 6,
        "NER": [f'["bananas", "flour", "sugar", "item{n}"]'
                for n in range(1, 6)] + ['["bananas", "oats"]'],
        "link": [f"{n}.com" for n in range(6)],
        "source": ["Gathered"] * 6,
    })
    settings = Settings(tfidf_min_df=1, tfidf_max_df=1.0,
                        artifacts_dir=tmp_path, processed_data_dir=tmp_path)
    return RecipeSearch(preprocess(raw), settings).fit()


def test_results_do_not_repeat_a_title(repeated_title_engine):
    titles = [r.title for r in repeated_title_engine.search("banana", top_k=5)]
    assert len(titles) == len(set(titles))


def test_the_cap_is_configurable(repeated_title_engine):
    repeated_title_engine.settings.max_results_per_title = 2
    titles = [r.title for r in repeated_title_engine.search("banana", top_k=5)]
    assert titles.count("Banana Bread") == 2


def test_cap_zero_returns_every_match(repeated_title_engine):
    results = repeated_title_engine.search("banana bread", top_k=5,
                                           max_per_title=0)
    assert [r.title for r in results].count("Banana Bread") > 1


def test_the_cap_keeps_the_highest_scoring_of_each_title(repeated_title_engine):
    uncapped = repeated_title_engine.search("banana bread", top_k=10,
                                            max_per_title=0)
    best = next(r for r in uncapped if r.title == "Banana Bread")
    capped = repeated_title_engine.search("banana bread", top_k=5)
    kept = next(r for r in capped if r.title == "Banana Bread")
    assert kept.recipe_id == best.recipe_id


def test_the_cap_backfills_rather_than_shrinking_the_page(repeated_title_engine):
    """Skipped rows are replaced from further down, not dropped."""
    results = repeated_title_engine.search("banana", top_k=2)
    assert len(results) == 2
    assert {r.title for r in results} == {"Banana Bread", "Banana Muffins"}


def test_results_stay_in_descending_score_order(repeated_title_engine):
    scores = [r.score for r in repeated_title_engine.search("banana", top_k=5)]
    assert scores == sorted(scores, reverse=True)


def test_pantry_matching_is_not_capped(repeated_title_engine):
    """Two recipes sharing a name need different things from a cupboard."""
    from src.search.pantry import PantryMatcher

    matches = PantryMatcher(repeated_title_engine,
                            repeated_title_engine.settings).match(
        have=["bananas", "flour", "sugar"], top_k=5)
    assert [m.title for m in matches].count("Banana Bread") > 1


# -- title-match bonus --------------------------------------------------

@pytest.fixture
def dish_name_engine(tmp_path) -> RecipeSearch:
    """The banana-bread problem, in miniature.

    A banana sandwich genuinely has banana and bread as *ingredients*,
    so the x3 NER weighting ranks it above a real banana bread whose
    NER lists eggs, sugar and butter. Only the title separates them.
    """
    raw = pd.DataFrame({
        "title": ["Banana Sandwich", "Banana Bread", "Lunchbox Sandwich"],
        "ingredients": [
            '["2 bananas", "2 slices bread", "peanut butter"]',
            '["3 eggs", "2 c. sugar", "1 c. butter", "baking soda"]',
            '["1 banana", "bread", "mayonnaise"]',
        ],
        "directions": ['["Assemble."]', '["Mash.", "Bake."]', '["Assemble."]'],
        "NER": [
            '["banana", "bread", "peanut butter"]',
            '["eggs", "sugar", "butter", "baking soda"]',
            '["banana", "bread", "mayonnaise"]',
        ],
        "link": ["a.com", "b.com", "c.com"],
        "source": ["Gathered"] * 3,
    })
    settings = Settings(tfidf_min_df=1, tfidf_max_df=1.0,
                        artifacts_dir=tmp_path, processed_data_dir=tmp_path)
    return RecipeSearch(preprocess(raw), settings).fit()


def rank_of(results, title: str) -> int:
    return next(i for i, r in enumerate(results) if r.title == title)


def test_without_the_bonus_an_ingredient_match_wins(dish_name_engine):
    """Documents the defect the title bonus exists to fix."""
    dish_name_engine.settings.title_match_bonus = 1.0
    results = dish_name_engine.search("banana bread", top_k=3)
    assert results[0].title != "Banana Bread"


def test_the_title_bonus_promotes_the_named_dish(dish_name_engine):
    """Asserts the direction, not an absolute rank.

    A three-document fixture has degenerate IDF -- every term is rare,
    so the ingredient signal is far stronger than in a real corpus and
    no sane bonus flips first place. On the 372k corpus the production
    value does put "Banana Bread" first; that is measured by
    evals/search_cases.json (case dish-01), not asserted here.
    """
    dish_name_engine.settings.title_match_bonus = 1.0
    before = dish_name_engine.search("banana bread", top_k=3)

    dish_name_engine.settings.title_match_bonus = 2.0
    after = dish_name_engine.search("banana bread", top_k=3)

    boosted = next(r for r in after if r.title == "Banana Bread")
    original = next(r for r in before if r.title == "Banana Bread")
    assert boosted.score > original.score
    assert rank_of(after, "Banana Bread") <= rank_of(before, "Banana Bread")


def test_only_the_title_matcher_is_boosted(dish_name_engine):
    """Recipes whose title lacks the query keep their exact score."""
    dish_name_engine.settings.title_match_bonus = 1.0
    before = {r.title: r.score for r in
              dish_name_engine.search("banana bread", top_k=3)}

    dish_name_engine.settings.title_match_bonus = 2.0
    after = {r.title: r.score for r in
             dish_name_engine.search("banana bread", top_k=3)}

    assert after["Lunchbox Sandwich"] == before["Lunchbox Sandwich"]
    assert after["Banana Sandwich"] == before["Banana Sandwich"]
    assert after["Banana Bread"] > before["Banana Bread"]


def test_the_bonus_does_not_disturb_ingredient_queries(dish_name_engine):
    """No title contains "banana peanut butter", so nothing is boosted."""
    with_bonus = dish_name_engine.search("banana peanut butter", top_k=3)
    dish_name_engine.settings.title_match_bonus = 1.0
    without = dish_name_engine.search("banana peanut butter", top_k=3)
    assert [r.title for r in with_bonus] == [r.title for r in without]


def test_the_bonus_cannot_rescue_an_unrelated_recipe(dish_name_engine):
    """It re-ranks candidates; it does not invent matches."""
    assert dish_name_engine.search("xylophone tungsten", top_k=3) == []


# -- similarity computation --------------------------------------------

def test_similarity_matches_cosine_similarity(engine):
    """The mat-vec form replaced linear_kernel for speed; it must not
    have changed the numbers."""
    from sklearn.metrics.pairwise import cosine_similarity

    query_vector = engine.vectorize_query("garlic chicken")
    expected = cosine_similarity(query_vector, engine.matrix).ravel()
    actual = (engine.matrix @ query_vector.T).toarray().ravel()
    assert actual == pytest.approx(expected, abs=1e-9)


def test_warm_populates_the_caches(engine):
    engine._title_cache = None
    engine._feature_name_cache = None
    engine.warm()
    assert engine._title_cache is not None
    assert engine._feature_name_cache is not None
