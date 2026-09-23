"""Central configuration.

Every tunable value lives here and is overridable from the environment
(or a local .env file), so no module hard-codes a path or a limit.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: settings.py -> config -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Dataset -------------------------------------------------------
    dataset_name: str = "mbien/recipe_nlg"
    dataset_split: str = "train"

    # RecipeNLG has ~2.2M rows. Loading all of them into pandas costs
    # several GB of RAM, which is pointless while prototyping, so we cap
    # it. Set MAX_RECIPES=0 to disable the cap.
    max_recipes: int = 50_000

    # The HF loader for recipe_nlg is a *manual download* dataset: it
    # cannot fetch the data itself. Point this at the directory holding
    # the extracted `full_dataset.csv` from recipenlg.cs.put.poznan.pl.
    dataset_dir: Path = PROJECT_ROOT / "data" / "raw"

    # --- Paths ---------------------------------------------------------
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    artifacts_dir: Path = PROJECT_ROOT / "data" / "artifacts"

    # --- Search ---------------------------------------------------------
    # Ignore terms in >60% of recipes ("salt", "water", "cup"): they carry
    # almost no discriminative signal but dominate the vocabulary.
    tfidf_max_df: float = 0.6
    # Ignore terms in fewer than 2 recipes: typos and one-off brand names.
    tfidf_min_df: int = 2
    # Unigrams + bigrams, so "olive oil" and "sour cream" survive as units.
    tfidf_ngram_max: int = 2
    tfidf_max_features: int = 50_000
    # Coverage boost: final score = cosine x coverage**weight, where
    # coverage is the fraction of the query's words the recipe contains.
    # Counteracts cosine's bias toward short recipes. 0 disables it.
    coverage_weight: float = 1.0
    # Pantry matching re-scores a TF-IDF candidate pool exactly. Bigger
    # pool = better recall, linear cost. 400 takes a few ms.
    pantry_candidate_pool: int = 400
    # Multiplier applied when the whole query appears in a recipe's
    # title. Dish-name queries ("banana bread") otherwise lose to
    # recipes that merely list those words as ingredients -- a banana
    # sandwich has NER [banana, bread], which the x3 NER weighting
    # scores above an actual banana bread whose NER is [eggs, sugar,
    # butter]. 1.0 disables the bonus.
    # 2.0 chosen by sweeping the value over evals/search_cases.json:
    # P@5 rises 0.785 -> 0.870 between 1.0 and 2.0 and is flat above it,
    # so this is the least aggressive value that reaches peak quality.
    title_match_bonus: float = 2.0
    # How many top candidates get the title bonus applied. The bonus
    # re-ranks, so it only needs to see recipes that could plausibly
    # reach the page.
    rerank_pool: int = 500
    # At most this many results may share a title. The corpus holds 100
    # recipes called "Banana Bread" -- genuinely different recipes, not
    # duplicates -- and without a cap a search for one returns six
    # near-identical-looking rows. 0 disables the cap.
    max_results_per_title: int = 1
    default_top_k: int = 5

    # --- LLM -----------------------------------------------------------
    # Off by default: everything works without a key, and turning the LLM
    # on should be a deliberate act, not an accident of the environment.
    llm_enabled: bool = False
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 2000
    llm_timeout_seconds: float = 30.0
    # Low effort: this is rephrasing supplied text, not solving anything.
    llm_effort: str = "low"

    # SecretStr keeps the key out of logs and repr(). Read from the
    # ANTHROPIC_API_KEY environment variable -- never written in code,
    # never committed. Leave unset to run without an LLM.
    anthropic_api_key: SecretStr | None = None

    # --- Services ------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    @property
    def processed_recipes_path(self) -> Path:
        return self.processed_data_dir / "recipes.parquet"

    @property
    def index_path(self) -> Path:
        """Fitted vectorizer + document matrix, pickled together."""
        return self.artifacts_dir / "tfidf_index.joblib"


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once per process."""
    return Settings()
