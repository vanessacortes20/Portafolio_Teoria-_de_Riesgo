"""
Tests unitarios del QP de Markowitz (modulo M6).

El QP se resuelve con cvxpy. Estos tests no requieren red y son rapidos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.logic import compute_efficient_frontier_qp, solve_markowitz_qp


def _toy_returns_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0010, 0.010, 252),
            "B": rng.normal(0.0012, 0.012, 252),
            "C": rng.normal(0.0008, 0.008, 252),
        }
    )


def test_qp_weights_sum_to_one_no_short():
    rets = _toy_returns_df()
    sol = solve_markowitz_qp(rets, target_return=None, allow_short=False)
    assert sol["feasible"] is True
    s = sum(sol["Weights"].values())
    assert abs(s - 1.0) < 1e-4


def test_qp_no_short_has_no_negative_weights():
    rets = _toy_returns_df()
    sol = solve_markowitz_qp(rets, target_return=None, allow_short=False)
    assert sol["feasible"] is True
    for w in sol["Weights"].values():
        assert w >= -1e-6


def test_qp_with_short_can_have_negative_weights():
    """No exigimos shorts; solo que la version con allow_short=True corra."""
    rets = _toy_returns_df()
    # Pedimos un rendimiento extremo (> max(mu)) para forzar shorts
    mu_annual = (rets.mean() * 252).values
    target = float(np.max(mu_annual)) * 1.5
    sol = solve_markowitz_qp(rets, target_return=target, allow_short=True)
    assert sol["feasible"] is True
    s = sum(sol["Weights"].values())
    assert abs(s - 1.0) < 1e-4


def test_efficient_frontier_has_points():
    rets = _toy_returns_df()
    res = compute_efficient_frontier_qp(rets, allow_short=False, n_points=20)
    assert res["feasible"] is True
    assert len(res["frontier"]) > 0
    # Min variance volatility <= cualquier punto de la frontera
    mv_vol = res["Min_Variance"]["Volatility"]
    for pt in res["frontier"]:
        assert pt["Volatility"] >= mv_vol - 1e-6
