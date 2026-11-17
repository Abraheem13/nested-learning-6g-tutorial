"""Discrete allocation templates over the slice simplex.

The action space of each agent is a finite set of PRB-allocation vectors
``a = (a_1, ..., a_S)`` with ``a_s >= 0`` and ``sum_s a_s = 1``.  Following the
system model, the simplex is discretised with a per-dimension granularity
``k`` so that every template has entries in ``{0, 1/k, 2/k, ..., 1}``.

The number of such templates is the number of weak compositions of ``k`` into
``S`` non-negative parts::

    K = C(k + S - 1, S - 1)

For the default configuration ``k = 4``, ``S = 3`` this gives ``K = C(6, 2) = 15``,
matching the 15 allocation templates of the reference system model.
"""

from __future__ import annotations

import itertools
from math import comb

import numpy as np

__all__ = ["simplex_templates", "n_templates"]


def n_templates(n_slices: int, granularity: int) -> int:
    """Number of discrete allocation templates ``K``."""
    return comb(granularity + n_slices - 1, n_slices - 1)


def simplex_templates(n_slices: int = 3, granularity: int = 4) -> np.ndarray:
    """Enumerate every allocation template on the discretised slice simplex.

    Parameters
    ----------
    n_slices:
        Number of slices ``S``.
    granularity:
        Per-dimension granularity ``k``; allocations are multiples of ``1/k``.

    Returns
    -------
    np.ndarray
        Array of shape ``(K, S)`` and dtype ``float32``.  Rows are sorted
        lexicographically so the enumeration is deterministic and stable
        across runs and platforms.
    """
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")
    if granularity < 1:
        raise ValueError("granularity must be >= 1")

    rows = []
    # Weak compositions of `granularity` into `n_slices` parts, via the
    # standard stars-and-bars bijection.  Deterministic ordering.
    for cuts in itertools.combinations(range(granularity + n_slices - 1), n_slices - 1):
        parts = []
        prev = -1
        for c in cuts:
            parts.append(c - prev - 1)
            prev = c
        parts.append(granularity + n_slices - 1 - prev - 1)
        rows.append(parts)

    table = np.asarray(rows, dtype=np.float64) / float(granularity)
    table = table[np.lexsort(table.T[::-1])]  # stable lexicographic order

    assert table.shape == (n_templates(n_slices, granularity), n_slices)
    assert np.allclose(table.sum(axis=1), 1.0)
    return np.ascontiguousarray(table, dtype=np.float32)
