"""TF-IDF + cosine-similarity recipe retrieval.

How it works
------------
**TF-IDF** turns each recipe's ``search_text`` into a vector of term
weights. Two factors multiply:

* *Term frequency* -- how often a term occurs in this recipe. This is
  where our field weighting from preprocessing pays off: NER terms were
  repeated three times, so they get three times the term frequency.
* *Inverse document frequency* -- ``log(N / documents containing the
  term)``. A term in every recipe ("salt", "cup") scores near zero; a
  term in a handful ("saffron", "tahini") scores high.

The product means a recipe is described by what makes it *unusual*, not
by what every recipe shares. Exactly right for ingredient matching:
"saffron" should decide a match, "water" should not.

**Cosine similarity** measures the angle between the query vector and
each recipe vector, ignoring their lengths. Length-independence matters
here because a three-word query and a forty-ingredient recipe have wildly
different magnitudes -- Euclidean distance would rank every short recipe
above every long one regardless of content. Cosine only asks "do these
point the same way", so it scores 0.0 (nothing in common) to 1.0
(identical term profile).

Since ``TfidfVectorizer`` L2-normalizes its output by default, every
vector already has length 1, so cosine similarity reduces to a plain dot
product. We use ``linear_kernel`` for that: same numbers as
``cosine_similarity``, without recomputing norms that are known to be 1.

Why this suits the prototype
----------------------------
* **No training, no GPU.** Fitting on 46k recipes takes seconds on a
  laptop; embeddings would need a model download and far more compute.
* **Fully explainable.** Every score traces back to specific shared
  terms, so when a result looks wrong you can see exactly why.
* **Strong on this task.** Recipe search is largely literal vocabulary
  overlap -- the user types ingredient nouns and the corpus contains
  those same nouns. Lexical matching is close to ideal for that.
* **Honest limits.** TF-IDF has no notion of meaning: "aubergine" will
  not match "eggplant", and "no dairy" matches recipes *containing*
  dairy, because the word is present either way. Those are the gaps an
  LLM or embedding layer fills later -- this class stays the retrieval
  core underneath it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger
from src.utils.text import normalize

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """One recommendation. Plain data, so the API and UI can both use it."""

    recipe_id: int
    title: str
    score: float                 # the ranking score: similarity x coverage
    similarity: float = 0.0      # raw cosine, before the coverage boost
    coverage: float = 1.0        # fraction of the query's words present
    ingredients: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    ner: list[str] = field(default_factory=list)
    link: str = ""
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RecipeSearch:
    """Fit a TF-IDF index over recipes, then answer queries against it.

    Usage::

        search = RecipeSearch.from_processed()      # loads parquet + fits
        results = search.search("chicken rice garlic", top_k=5)

    Or, once an index has been built and saved::

        search = RecipeSearch.load()                # seconds, not minutes
    """

    TEXT_COLUMN = "search_text"

    def __init__(
        self,
        recipes: pd.DataFrame,
        settings: Settings | None = None,
        vectorizer: TfidfVectorizer | None = None,
        matrix=None,
    ):
        if self.TEXT_COLUMN not in recipes.columns:
            raise KeyError(
                f"recipes must have a '{self.TEXT_COLUMN}' column; "
                "run src.data.preprocess.preprocess first"
            )
        self.settings = settings or get_settings()
        self.recipes = recipes.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.matrix = matrix
        # Built lazily by _feature_names(); get_feature_names_out()
        # allocates a 50k-element array on every call, which otherwise
        # costs more than the similarity computation itself.
        self._feature_name_cache = None
        # Normalized titles, for the title-match bonus. Computed once.
        self._title_cache = None

    # -- construction ---------------------------------------------------

    @classmethod
    def from_processed(cls, settings: Settings | None = None) -> "RecipeSearch":
        """Build from data/processed/recipes.parquet and fit immediately."""
        from src.data.preprocess import load_processed

        settings = settings or get_settings()
        engine = cls(load_processed(settings), settings)
        engine.fit()
        return engine.warm()

    def fit(self) -> "RecipeSearch":
        """Learn the vocabulary and build the recipe-term matrix."""
        s = self.settings
        self.vectorizer = TfidfVectorizer(
            # search_text is already lowercased and punctuation-free, so
            # the vectorizer's own preprocessing would be wasted work.
            lowercase=False,
            ngram_range=(1, s.tfidf_ngram_max),
            max_df=s.tfidf_max_df,
            min_df=s.tfidf_min_df,
            max_features=s.tfidf_max_features,
            sublinear_tf=True,
            # Sublinear TF (1 + log tf) stops a recipe that lists
            # "chicken" eight times from burying one that lists it twice;
            # the eighth mention is not four times more relevant.
        )
        self.matrix = self.vectorizer.fit_transform(
            self.recipes[self.TEXT_COLUMN].fillna("")
        )
        self._feature_name_cache = None
        self._title_cache = None
        logger.info(
            "Fitted TF-IDF: %d recipes x %d terms (%.1f MB sparse)",
            *self.matrix.shape,
            self.matrix.data.nbytes / 1e6,
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return self.vectorizer is not None and self.matrix is not None

    # -- querying -------------------------------------------------------

    def vectorize_query(self, query: str):
        """Project a user query into the same vector space as the recipes.

        ``transform`` (not ``fit_transform``): the query must use the
        vocabulary and IDF weights learned from the corpus, otherwise its
        coordinates mean something different from the recipes' and the
        similarity scores are nonsense. Terms the corpus never saw are
        silently dropped -- there is nothing to match them against.
        """
        self._require_fitted()
        return self.vectorizer.transform([normalize(query)])

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float = 0.0,
        max_per_title: int | None = None,
    ) -> list[SearchResult]:
        """Return the ``top_k`` most similar recipes, best first.

        ``max_per_title`` limits how many results may share a title.
        This is a presentation concern, not a data one: the corpus has
        100 distinct recipes called "Banana Bread", so a search for it
        legitimately matches all of them and fills the page with what
        looks like the same answer repeated. Pass 0 or None to disable
        -- the pantry matcher does, because it wants a wide candidate
        pool and dedups nothing.
        """
        self._require_fitted()
        top_k = top_k or self.settings.default_top_k

        query_vector = self.vectorize_query(query)
        if query_vector.nnz == 0:
            # Nothing in the query is in the vocabulary. Every score would
            # be 0.0; returning [] is more honest than five arbitrary rows.
            logger.info("Query %r has no known terms", query)
            return []

        # matrix @ query.T, not linear_kernel(query, matrix): the latter
        # computes safe_sparse_dot(query, matrix.T), and transposing a
        # 372k x 50k CSR on every request cost 205ms of a 240ms query.
        # This form is a plain sparse mat-vec -- identical numbers, 6x
        # faster. Both vectors are L2-normalized, so the dot product is
        # the cosine similarity.
        similarity = (self.matrix @ query_vector.T).toarray().ravel()
        coverage = self._coverage(query_vector)
        scores = similarity * (coverage ** self.settings.coverage_weight)

        scores = self._apply_title_bonus(scores, query)

        cap = (self.settings.max_results_per_title
               if max_per_title is None else max_per_title)

        # With a cap, scan deeper than top_k: the winners we skip have
        # to be replaced by something further down the ranking.
        depth = top_k if not cap else min(top_k * 20 + 50, scores.size)

        results: list[SearchResult] = []
        seen: dict[str, int] = {}
        for index in self._top_indices(scores, depth):
            if scores[index] <= min_score:
                break            # scores descend; nothing later qualifies
            if cap:
                title = normalize(str(self.recipes.iloc[index].get("title", "")))
                if seen.get(title, 0) >= cap:
                    continue
                seen[title] = seen.get(title, 0) + 1
            results.append(
                self._build_result(index, float(scores[index]), query_vector,
                                   similarity=float(similarity[index]),
                                   coverage=float(coverage[index]))
            )
            if len(results) == top_k:
                break
        return results

    def warm(self) -> "RecipeSearch":
        """Pay the one-off caches at startup instead of on a user query.

        Normalizing 372k titles takes ~450ms. Left lazy, the first
        person to search waits for it -- which showed up as a p95 eight
        times the median while the median looked fine.
        """
        self._titles()
        self._feature_names()
        return self

    def _titles(self):
        if self._title_cache is None:
            self._title_cache = self.recipes["title"].map(normalize)
        return self._title_cache

    def _apply_title_bonus(self, scores: np.ndarray, query: str) -> np.ndarray:
        """Promote recipes whose title contains the whole query.

        An exact title containment is strong evidence the user named a
        dish rather than listed ingredients, and it is the only signal
        that separates "Banana Bread" from "Banana Sandwich" -- the
        sandwich genuinely contains banana and bread, so no amount of
        ingredient weighting will do it.

        Applied to the top `rerank_pool` candidates only. The bonus
        re-orders; it cannot rescue a recipe that had no lexical overlap
        to begin with, so candidates outside the pool could not reach
        the page anyway. Ingredient queries are unaffected -- "chicken
        tomato onion" appears in no title.
        """
        bonus = self.settings.title_match_bonus
        needle = normalize(query)
        if bonus <= 1.0 or not needle:
            return scores

        pool = min(self.settings.rerank_pool, scores.size)
        candidates = np.argpartition(-scores, pool - 1)[:pool]
        titles = self._titles()

        boosted = scores.copy()
        for index in candidates:
            if needle in titles.iat[index]:
                boosted[index] *= bonus
        return boosted

    def _coverage(self, query_vector) -> np.ndarray:
        """Fraction of the query's known words each recipe contains.

        The denominator is the query's in-vocabulary unigrams, not every
        word the user typed. A term the corpus has never seen cannot be
        matched by any recipe, so counting it would penalise all of them
        equally and change nothing but the printed number.

        Why this exists: cosine similarity divides by vector length, so a
        three-ingredient guacamole that matches two of three query words
        scores higher than an eleven-ingredient cacciatore that matches
        all three. The scores are correct; the objective is wrong. A
        recipe search should prefer having *all* your ingredients.

        Only unigrams count. Bigram features ("tomato onion") are
        word-order accidents -- a recipe listing "onion, tomato" has the
        same ingredients and should not be penalised for the ordering.

        Returns an all-ones vector when the weight is 0 or the query has
        no unigrams, which makes the boost a no-op rather than a crash.
        """
        rows = self.matrix.shape[0]
        if self.settings.coverage_weight == 0:
            return np.ones(rows)

        names = self._feature_names()
        unigrams = [i for i in query_vector.indices if " " not in names[i]]
        if not unigrams:
            return np.ones(rows)

        # Column slice of the document matrix, then count non-zeros per
        # row: how many of the query's words this recipe contains.
        present = self.matrix[:, unigrams].getnnz(axis=1)
        return present / len(unigrams)

    @staticmethod
    def _top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
        """Indices of the top_k scores, sorted best-first.

        ``argpartition`` is O(n) and only orders the k winners, versus
        ``argsort``'s O(n log n) full ordering of 46k rows we discard.
        """
        top_k = min(top_k, scores.size)
        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        return candidates[np.argsort(-scores[candidates])]

    @staticmethod
    def as_list(value) -> list[str]:
        """Coerce a cell to a plain list of strings.

        Parquet round-trips list columns as numpy arrays, and truth-testing
        an array raises. So test for None explicitly, never with `or`.
        """
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return [str(v) for v in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)]

    def _build_result(self, index: int, score: float, query_vector,
                      similarity: float = 0.0,
                      coverage: float = 1.0) -> SearchResult:
        row = self.recipes.iloc[index]
        link = row.get("link")
        return SearchResult(
            recipe_id=int(row.get("recipe_id", index)),
            title=str(row.get("title", "")),
            score=round(score, 4),
            similarity=round(similarity, 4),
            coverage=round(coverage, 4),
            ingredients=self.as_list(row.get("ingredients")),
            directions=self.as_list(row.get("directions")),
            ner=self.as_list(row.get("ner")),
            link="" if link is None or pd.isna(link) else str(link),
            matched_terms=self._matched_terms(index, query_vector),
        )

    def _matched_terms(self, index: int, query_vector) -> list[str]:
        """Terms shared by query and recipe -- the "why" behind a score.

        Cheap to compute (intersect two sparse rows) and it makes results
        debuggable without reading the vectorizer internals.
        """
        query_terms = set(query_vector.indices)
        recipe_terms = set(self.matrix[index].indices)
        shared = query_terms & recipe_terms
        if not shared:
            return []
        names = self._feature_names()
        return sorted(names[i] for i in shared)

    def _feature_names(self):
        if self._feature_name_cache is None:
            self._feature_name_cache = self.vectorizer.get_feature_names_out()
        return self._feature_name_cache

    # -- persistence ----------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Persist vectorizer + matrix so the API starts fast.

        The recipe table is not pickled with them -- it already lives in
        parquet, and duplicating it would double the artifact size and
        let the two copies drift apart.
        """
        import joblib

        self._require_fitted()
        path = path or self.settings.index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix}, path)
        logger.info("Saved index to %s", path)
        return path

    @classmethod
    def load(
        cls, path: Path | None = None, settings: Settings | None = None
    ) -> "RecipeSearch":
        import joblib

        from src.data.preprocess import load_processed

        settings = settings or get_settings()
        path = path or settings.index_path
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m scripts.build_index"
            )
        payload = joblib.load(path)
        recipes = load_processed(settings)
        engine = cls(
            recipes,
            settings,
            vectorizer=payload["vectorizer"],
            matrix=payload["matrix"],
        )
        if engine.matrix.shape[0] != len(recipes):
            raise ValueError(
                f"Index has {engine.matrix.shape[0]} rows but "
                f"{len(recipes)} recipes are on disk -- rebuild the index."
            )
        logger.info("Loaded index from %s (%d recipes)", path, len(recipes))
        return engine.warm()

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() (or use from_processed/load) first")
