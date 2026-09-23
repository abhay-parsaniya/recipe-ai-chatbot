"""Dataset loading. Covers the CSV path and the hub fallback's failure
mode -- previously 0% covered, which meant a fresh clone's very first
command was the least-tested code in the project.
"""

import pandas as pd
import pytest

from src.config.settings import Settings
from src.data.loader import (
    CSV_CANDIDATES,
    find_local_csv,
    load_from_csv,
    load_raw_recipes,
)

HEADER = ",title,ingredients,directions,link,source,NER\n"
ROW = ('{i},Recipe {i},"[""1 c. flour""]","[""Mix.""]",'
       'x.com,Gathered,"[""flour""]"\n')


def write_csv(path, rows: int = 5) -> None:
    path.write_text(HEADER + "".join(ROW.format(i=i) for i in range(rows)))


def test_finds_the_canonical_filename(tmp_path):
    target = tmp_path / CSV_CANDIDATES[0]
    write_csv(target)
    assert find_local_csv(tmp_path) == target


def test_finds_a_lone_csv_under_any_name(tmp_path):
    target = tmp_path / "whatever.csv"
    write_csv(target)
    assert find_local_csv(tmp_path) == target


def test_two_ambiguous_csvs_are_not_guessed_between(tmp_path):
    write_csv(tmp_path / "a.csv")
    write_csv(tmp_path / "b.csv")
    assert find_local_csv(tmp_path) is None


def test_empty_directory_finds_nothing(tmp_path):
    assert find_local_csv(tmp_path) is None


def test_load_from_csv_drops_the_unnamed_index_column(tmp_path):
    path = tmp_path / "full_dataset.csv"
    write_csv(path, rows=3)
    frame = load_from_csv(path)
    assert "Unnamed: 0" not in frame.columns
    assert list(frame.columns) == ["title", "ingredients", "directions",
                                   "link", "source", "NER"]


def test_limit_is_applied_at_read_time(tmp_path):
    path = tmp_path / "full_dataset.csv"
    write_csv(path, rows=20)
    assert len(load_from_csv(path, limit=5)) == 5


def test_load_raw_recipes_prefers_the_local_csv(tmp_path):
    write_csv(tmp_path / "full_dataset.csv", rows=7)
    settings = Settings(dataset_dir=tmp_path, max_recipes=0)
    assert len(load_raw_recipes(settings=settings)) == 7


def test_max_recipes_setting_caps_the_load(tmp_path):
    write_csv(tmp_path / "full_dataset.csv", rows=20)
    settings = Settings(dataset_dir=tmp_path, max_recipes=4)
    assert len(load_raw_recipes(settings=settings)) == 4


def test_missing_data_gives_the_download_instructions(tmp_path):
    """The error a new user hits first must say what to do about it."""
    settings = Settings(dataset_dir=tmp_path / "nothing-here", max_recipes=10)
    with pytest.raises(FileNotFoundError) as excinfo:
        load_raw_recipes(settings=settings)

    message = str(excinfo.value)
    assert "recipenlg.cs.put.poznan.pl" in message
    assert "full_dataset.csv" in message


def test_loaded_frame_feeds_preprocess(tmp_path):
    """The loader's output shape is what preprocess expects."""
    from src.data.preprocess import preprocess

    write_csv(tmp_path / "full_dataset.csv", rows=3)
    raw = load_raw_recipes(settings=Settings(dataset_dir=tmp_path,
                                             max_recipes=0))
    clean = preprocess(raw)
    assert list(clean["title"])[0] == "Recipe 0"
    assert clean["ingredients"].iloc[0] == ["1 c. flour"]


def test_limit_zero_means_everything_not_the_default_cap(tmp_path):
    """Regression: `--limit 0` used to collapse to None and silently
    re-apply MAX_RECIPES, so the full corpus was unreachable."""
    write_csv(tmp_path / "full_dataset.csv", rows=30)
    settings = Settings(dataset_dir=tmp_path, max_recipes=10)

    assert len(load_raw_recipes(limit=0, settings=settings)) == 30
    assert len(load_raw_recipes(limit=None, settings=settings)) == 10
    assert len(load_raw_recipes(limit=7, settings=settings)) == 7
