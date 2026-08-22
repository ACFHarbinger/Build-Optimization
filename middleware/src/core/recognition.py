"""
RecognizedName seam for the STS2 reward advisor's recognition pipeline (SA10).

This module is the *contract* between the OCR/ingestion layer (SA8, which
transcribes a screenshot crop to raw text) and the advisor's confirmed-`Item`
world (SA1/SA3/SA4). Recognition must never silently become a card: raw OCR
text only becomes a ``matched_card_id`` when the match is unambiguous. Any
ambition to treat a screenshot as ground truth is deliberately rejected here.

The seam keeps the decision *deterministic and dataset-owned*: normalize, then
exact canonical name, then approved aliases, then a bounded fuzzy candidate
list. ``+`` is preserved because an upgraded card is a different card. An
unmatched name yields ``needs_dataset_entry=True`` (and whatever the user
manually picks is ``method="manual"``); low-confidence names are never
silently selected.

Per Harbinger's product call, the *uncertainty policy* is a user setting: at
or above a chosen confidence threshold, return a flagged tentative answer;
below it, block for correction. The default threshold lives in the
``recognition`` Hydra config group (SA8/SA10 wiring), not here.

This module is deliberately core-only: it imports nothing from
``backend/``/``pipeline``/``ui``/``frontend``, and no OCR/vision/torch
library. It operates on plain text; it does not read images.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class MatchMethod(str, Enum):
    """How a recognised name was resolved to a catalogue card."""

    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy"
    CLOUD = "cloud"
    MANUAL = "manual"


class ConfidencePolicy(str, Enum):
    """Per-user uncertainty policy applied to a recognition result."""

    # Since at least one match method always "counts" toward acceptance, this
    # is expressed as an outcome of applying a threshold, not an input.
    TENTATIVE = "tentative"
    BLOCK = "block"


@dataclass(frozen=True)
class RecognizedName:
    """One recognised card name, kept as an explicit reviewable record.

    Attributes:
        region_id: Identifies the offer/deck-grid region this came from.
        raw_text:  The exact OCR or manually-entered text, verbatim.
        normalized: Case/Unicode/punctuation-normalised candidate name.
        confidence: [0, 1] unit-interval confidence in the transcription.
        method:    CRAN-match resolution method.
        matched_card_id: Catalogue card id, only when the match is unambiguous.
        needs_dataset_entry: True when the name does not resolve — the advisor
            must block until the user adds/overrides a catalogue row.
        candidates: Bounded list of (card_id, similarity) for display when
            unambiguous resolution is impossible; never silently auto-picked.
    """

    region_id: str
    raw_text: str
    normalized: str
    confidence: float
    method: MatchMethod = MatchMethod.EXACT
    matched_card_id: Optional[str] = None
    needs_dataset_entry: bool = False
    candidates: Tuple[Tuple[str, float], ...] = ()

    def is_resolved(self) -> bool:
        """True when a card was matched with confidence at accepted level."""
        return self.matched_card_id is not None and not self.needs_dataset_entry


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_name(raw: str) -> str:
    """Normalise a card name to a canonical comparison string.

    Strips case, folds Unicode (so a full-width "＋" becomes "+"), collapses
    internal whitespace, and removes leading/trailing punctuation. The ``+``
    upgrade marker is *preserved* — ``Carnage`` and ``Carnage+`` must stay
    distinct → distinct ids.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = re.sub(r"[^\w\s+]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ---------------------------------------------------------------------------
# Fuzzy similarity (bounded, no external dep)
# ---------------------------------------------------------------------------


def similarity_score(a: str, b: str) -> float:
    """Normalised [0, 1] edit-similarity between two already-normalised names.

    Uses a bounded Levenshtein (banded by length difference) so it is cheap
    and deterministic. 1.0 means identical; 0.0 means disjoint.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if abs(len(a) - len(b)) > max(len(a), len(b)):
        return 0.0

    # Standard two-row Levenshtein DP.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[len(b)]
    return 1.0 - dist / max(len(a), len(b))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class NameMatcher:
    """Deterministic, dataset-owned resolution of a raw name to a card id.

    Resolution order (first unambiguous hit wins):
      1. exact canonical name (after normalisation),
      2. approved aliases (also normalised),
      3. a bounded fuzzy candidate list — returned as candidates, but a single
         clear-best past a similarity floor may be auto-matched; a near tie is
         never silently picked.

    Args:
        catalogue: Sequence of ``Item``-like card ids/names. The caller (e.g.
            the SA8 pipeline) is responsible for loading the catalogue as the
            same authoritative source the advisor reads; this matcher just
            needs ``(card_id, display_name)`` pairs.
        aliases: Optional mapping ``card_id -> list[alias]``.
        fuzzy_threshold: Minimum similarity to auto-match fuzzily (default
            0.90 — conservative, never a near tie).
        top_k: Bounded size of the fuzzy candidate list (default 3).
    """

    def __init__(
        self,
        catalogue: Sequence[Tuple[str, str]],
        aliases: Optional[Dict[str, Sequence[str]]] = None,
        fuzzy_threshold: float = 0.90,
        top_k: int = 3,
    ) -> None:
        self._by_name: Dict[str, str] = {}
        self._by_alias: Dict[str, str] = {}
        self._names: List[Tuple[str, str, str]] = []  # (card_id, norm, compact)

        for card_id, display in catalogue:
            norm = normalize_name(display)
            self._by_name[norm] = card_id
            self._names.append((card_id, norm, norm.replace(" ", "")))
            for alias in (aliases or {}).get(card_id, []):
                self._by_alias[normalize_name(alias)] = card_id

        self.fuzzy_threshold = fuzzy_threshold
        self.top_k = top_k

    def resolve(self, raw: str) -> Tuple[Optional[str], MatchMethod, Tuple[Tuple[str, float], ...]]:
        """Resolve ``raw`` to ``(card_id, method, candidates)``.

        ``card_id`` is ``None`` unless exact/alias/fuzzy-clear resolve, in
        which case ``method`` reflects how it was matched. ``candidates`` is
        never empty when an unambiguous match wasn't formed, so the caller can
        present them for a manual pick.
        """
        norm = normalize_name(raw)

        exact = self._by_name.get(norm)
        if exact is not None:
            return exact, MatchMethod.EXACT, ()

        alias = self._by_alias.get(norm)
        if alias is not None:
            return alias, MatchMethod.ALIAS, ()

        ranked = self._rank_fuzzy(norm)
        if not ranked:
            return None, MatchMethod.FUZZY, ()

        best_id, best_sim = ranked[0]
        if best_sim >= self.fuzzy_threshold and (len(ranked) == 1 or ranked[1][1] < best_sim - 1e-9):
            return best_id, MatchMethod.FUZZY, tuple(ranked)
        return None, MatchMethod.FUZZY, tuple(ranked)

    def _rank_fuzzy(self, norm: str) -> List[Tuple[str, float]]:
        """Bound, descending-similarity list of (card_id, similarity).

        Comparison is **space-insensitive**: OCR frequently inserts or drops a
        space ("Inf lame" ← "Inflame"), so each candidate is matched against
        both the normalised name and its compact (space-stripped) form, taking
        the better score.
        """
        compact = norm.replace(" ", "")
        ranked = [
            (card_id, max(similarity_score(norm, candidate_norm), similarity_score(compact, candidate_compact)))
            for card_id, candidate_norm, candidate_compact in self._names
        ]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked[: self.top_k]


# ---------------------------------------------------------------------------
# Confidence policy
# ---------------------------------------------------------------------------


def apply_confidence(
    name: RecognizedName,
    accept_threshold: float,
) -> ConfidencePolicy:
    """Decide the per-user uncertainty outcome for a recognition.

    At/above ``accept_threshold`` and resolved → TENTATIVE (a flagged answer
    the user may accept). Otherwise → BLOCK (require correction). A resolved
    *exact/alias* match is accepted regardless of the threshold (there is no
    transcription uncertainty to resolve). An unresolved name always blocks.
    """
    if name.needs_dataset_entry or name.matched_card_id is None:
        return ConfidencePolicy.BLOCK
    if name.method in (MatchMethod.EXACT, MatchMethod.ALIAS):
        return ConfidencePolicy.TENTATIVE
    return ConfidencePolicy.TENTATIVE if name.confidence >= accept_threshold else ConfidencePolicy.BLOCK


def resolve_name(
    matcher: NameMatcher,
    region_id: str,
    raw_text: str,
    confidence: float,
    accept_threshold: float,
) -> Tuple[RecognizedName, ConfidencePolicy]:
    """Convenience: run the matcher, build a RecognizedName, apply the policy.

    Returns the record and the resulting outcome. This is the single seam the
    SA8 transcription layer calls after OCR-ing a crop.
    """
    card_id, method, candidates = matcher.resolve(raw_text)
    needs_entry = card_id is None
    name = RecognizedName(
        region_id=region_id,
        raw_text=raw_text,
        normalized=normalize_name(raw_text),
        confidence=max(0.0, min(1.0, confidence)),
        method=method,
        matched_card_id=card_id,
        needs_dataset_entry=needs_entry,
        candidates=candidates,
    )
    return name, apply_confidence(name, accept_threshold)
