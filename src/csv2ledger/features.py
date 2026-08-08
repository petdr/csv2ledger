"""Vectorise descriptions for nearest-neighbour lookup.

HashingVectorizer is used deliberately in place of TfidfVectorizer. A hashing vectoriser
has no fitted vocabulary, so a newly confirmed transaction can be appended to the
training matrix immediately, without refitting and without the feature space shifting
underneath the rows already stored. That is what makes the learning genuinely
incremental rather than a periodic retrain.

Three things are vectorised, and each earns its place:

  * character n-grams over the whole description -- the layer that generalises, since
    they let "WW METRO 8281 MELBOURNE VI" sit close to "WW METRO 1234 CARLTON VI"
  * a separate block over the leading characters -- card-network descriptions are
    fixed-width with the merchant first and the suburb second, so without this the
    shared "MELBOURNE VI" tail pulls unrelated merchants together
  * bank and debit/credit direction -- weak tie-breakers only

IDF weights are fitted once over the seed corpus and then held fixed. Document
frequencies barely move when a few hundred rows are added to six thousand, and freezing
them is what keeps appending a row cheap: new rows are weighted with the existing IDF
rather than invalidating every vector already in the index.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize as l2_normalize

CHAR_FEATURES = 2**18
PREFIX_FEATURES = 2**16
WORD_FEATURES = 2**16
CONTEXT_FEATURES = 2**8

# Number of leading characters treated as the merchant name. Card-network descriptions
# pad the merchant into a fixed-width field roughly this long.
PREFIX_LENGTH = 24

# Relative weights. The prefix block is boosted because the merchant name is the signal;
# context is kept small so it can break ties without ever outvoting the description.
PREFIX_WEIGHT = 1.5
CONTEXT_WEIGHT = 0.15


class FeatureSpace:
    """Maps (description, bank, direction) triples into a shared vector space."""

    def __init__(self) -> None:
        self._char = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=CHAR_FEATURES,
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )
        self._prefix = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=PREFIX_FEATURES,
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )
        self._word = HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            n_features=WORD_FEATURES,
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )
        self._context = HashingVectorizer(
            analyzer="word",
            n_features=CONTEXT_FEATURES,
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )
        self._idf: TfidfTransformer | None = None

    @property
    def width(self) -> int:
        return CHAR_FEATURES + PREFIX_FEATURES + WORD_FEATURES + CONTEXT_FEATURES

    def _raw(
        self, descriptions: list[str], banks: list[str], directions: list[str]
    ) -> sparse.csr_matrix:
        context = [f"bank_{b} dir_{d}" for b, d in zip(banks, directions, strict=True)]
        prefixes = [d[:PREFIX_LENGTH] for d in descriptions]
        blocks = [
            self._char.transform(descriptions),
            self._prefix.transform(prefixes) * PREFIX_WEIGHT,
            self._word.transform(descriptions),
            self._context.transform(context) * CONTEXT_WEIGHT,
        ]
        return sparse.hstack(blocks, format="csr", dtype=np.float64)

    def fit(
        self,
        descriptions: list[str],
        banks: list[str] | None = None,
        directions: list[str] | None = None,
    ) -> None:
        """Learn IDF weights from the seed corpus.

        Down-weights the boilerplate that appears in most descriptions -- suburb and
        state suffixes, "Visa Purchase", "EFTPOS" -- so similarity is driven by the
        merchant name rather than by shared formatting.
        """
        if not descriptions:
            return
        banks = banks or [""] * len(descriptions)
        directions = directions or [""] * len(descriptions)
        transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
        transformer.fit(self._raw(descriptions, banks, directions))
        self._idf = transformer

    def transform(
        self,
        descriptions: list[str],
        banks: list[str] | None = None,
        directions: list[str] | None = None,
    ) -> sparse.csr_matrix:
        """Vectorise and L2-normalise, so a dot product is a cosine similarity."""
        if not descriptions:
            return sparse.csr_matrix((0, self.width), dtype=np.float64)

        banks = banks or [""] * len(descriptions)
        directions = directions or [""] * len(descriptions)
        matrix = self._raw(descriptions, banks, directions)
        if self._idf is not None:
            matrix = self._idf.transform(matrix)
        return l2_normalize(matrix, norm="l2", axis=1, copy=False)

    def transform_one(
        self, description: str, bank: str = "", direction: str = ""
    ) -> sparse.csr_matrix:
        return self.transform([description], [bank], [direction])


def cosine_scores(query: sparse.csr_matrix, matrix: sparse.csr_matrix) -> np.ndarray:
    """Cosine similarity of one query row against every row of ``matrix``.

    Both sides are already L2-normalised, so the dot product is the cosine directly.
    """
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray((matrix @ query.T).todense()).ravel()
