import pandas as pd
import pytest

from src.data.preprocess import build_search_text, preprocess
from src.utils.text import normalize, parse_list_field


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": ["Banana Bread", "Banana Bread", "", "Chili"],
            "ingredients": [
                '["2 c. flour", "3 bananas"]',
                '["2 c. flour", "3 bananas"]',
                '["1 egg"]',
                '["1 lb. beef", "2 cans beans"]',
            ],
            "directions": [
                '["Mix.", "Bake 60 minutes."]',
                '["Mix.", "Bake."]',
                '["Fry."]',
                '["Brown the beef.", "Simmer."]',
            ],
            "NER": [
                '["flour", "bananas"]',
                '["flour", "bananas"]',
                '["egg"]',
                '["beef", "beans"]',
            ],
            "link": ["a.com", "b.com", "c.com", "d.com"],
            "source": ["Gathered"] * 4,
        }
    )


def test_parse_list_field_handles_bad_rows():
    assert parse_list_field('["a", "b"]') == ["a", "b"]
    assert parse_list_field("not a list") == ["not a list"]
    assert parse_list_field(None) == []
    assert parse_list_field("") == []


def test_normalize_strips_punctuation_and_case():
    assert normalize("2 c. Flour, sifted!") == "2 c flour sifted"


def test_preprocess_drops_blanks_and_duplicates():
    out = preprocess(raw_frame())
    # blank title dropped, duplicate Banana Bread collapsed
    assert list(out["title"]) == ["Banana Bread", "Chili"]
    assert list(out["recipe_id"]) == [0, 1]
    assert out["ingredients"].iloc[0] == ["2 c. flour", "3 bananas"]


def test_search_text_weights_ner_highest():
    out = preprocess(raw_frame())
    tokens = out["search_text"].iloc[0].split()
    # "flour" is in NER (x3) and in the ingredient line (x1)
    assert tokens.count("flour") == 4
    # "bread" only appears in the title (x2)
    assert tokens.count("bread") == 2
    assert "bake" not in tokens  # directions excluded


def test_build_search_text_ignores_missing_fields():
    row = pd.Series({"title": "Soup", "ner": ["water"]})
    assert build_search_text(row) == "water water water soup soup"


def test_preprocess_requires_core_columns():
    with pytest.raises(KeyError):
        preprocess(pd.DataFrame({"title": ["x"]}))


def test_reprints_are_dropped_but_namesakes_are_kept():
    """Same title + same ingredients is a reprint; same title alone is not.

    The corpus has 100 distinct recipes called "Banana Bread". Keying
    dedup on the title would delete 99 real recipes.
    """
    raw = pd.DataFrame({
        "title": ["Banana Bread", "Banana Bread", "Banana Bread"],
        "ingredients": [
            '["2 bananas", "1 c. flour"]',
            '["1 c. flour", "2 bananas"]',   # same set, different order
            '["5 bananas", "4 c. flour"]',   # genuinely different recipe
        ],
        "directions": ['["Bake."]', '["Bake."]', '["Bake longer."]'],
        "NER": ['["bananas", "flour"]', '["flour", "bananas"]',
                '["bananas", "flour", "walnuts"]'],
        "link": ["a.com", "b.com", "c.com"],
        "source": ["Gathered"] * 3,
    })
    out = preprocess(raw)
    assert len(out) == 2
    assert out["ingredients"].iloc[1] == ["5 bananas", "4 c. flour"]


def test_ingredient_order_does_not_defeat_dedup():
    """Order and casing are ignored; wording is not.

    "1 c. sugar" and "1 cup sugar" stay distinct -- unifying units would
    need a measurement parser, and the NER pass catches those reprints.
    """
    from src.data.preprocess import _ingredient_fingerprint

    assert (_ingredient_fingerprint(["1 c. Sugar", "2 Eggs"])
            == _ingredient_fingerprint(["2 eggs", "1 c. sugar"]))
    assert (_ingredient_fingerprint(["1 c. sugar"])
            != _ingredient_fingerprint(["1 cup sugar"]))


def test_apostrophes_are_deleted_not_split():
    """Regression: "Shepherd's" used to normalize to "shepherd s", so a
    user typing "shepherds pie" matched none of the 226 Shepherd's Pie
    recipes. The two spellings must land on the same token."""
    from src.utils.text import normalize

    assert normalize("Shepherd's Pie") == "shepherds pie"
    assert normalize("shepherds pie") == "shepherds pie"
    assert normalize("Jewell Ball'S Chicken") == "jewell balls chicken"
    assert normalize("don't") == "dont"
