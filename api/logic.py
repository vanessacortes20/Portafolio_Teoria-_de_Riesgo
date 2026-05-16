from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# SECCIÓN 1: ANÁLISIS TÉCNICO (MÓDULO 1)
# =============================================================================

def calculate_sma(data: pd.DataFrame, window: int = 20) -> pd.Series:
    return data["Close"].rolling(window=window).mean()


def calculate_ema(data: pd.DataFrame, window: int = 20) -> pd.Series:
    return data["Close"].ewm(span=window, adjust=False).mean()


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data: pd.DataFrame):
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    data: pd.DataFrame, window: int = 20, num_std: int = 2
):
    sma = calculate_sma(data, window)
    std = data["Close"].rolling(window=window).std()
    return sma + (std * num_std), sma - (std * num_std)


def calculate_stochastic(
    data: pd.DataFrame, k_window: int = 14, d_window: int = 3
):
    low_min = data["Low"].rolling(window=k_window).min()
    high_max = data["High"].rolling(window=k_window).max()
    k = 100 * (data["Close"] - low_min) / (high_max - low_min)
    return k, k.rolling(window=d_window).mean()


# =============================================================================
# SECCIÓN 2: RENDIMIENTOS (MÓDULO 2)
# =============================================================================

def calculate_returns(data: pd.DataFrame):
    simple_ret = data["Close"].pct_change().dropna()
    log_ret = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    return simple_ret, log_ret


def get_descriptive_stats(returns: pd.Series) -> dict:
    return {
        "Media": returns.mean(),
        "Desviación Estándar": returns.std(),
        "Asimetría (Skewness)": returns.skew(),
        "Curtosis": returns.kurtosis(),
        "Mínimo": returns.min(),
        "Máximo": returns.max(),
        "Conteo": len(returns),
    }


def perform_normality_tests(returns: pd.Series) -> dict:
    jb_stat, jb_p = stats.jarque_bera(returns)
    sw_stat, sw_p = stats.shapiro(returns[:5000])
    return {
        "Jarque-Bera": {"stat": jb_stat, "p_value": jb_p},
        "Shapiro-Wilk": {"stat": sw_stat, "p_value": sw_p},
    }


# =============================================================================
# SECCIÓN 3: MODELOS ARCH/GARCH (MÓDULO 3)
# =============================================================================
from arch import arch_model


def fit_garch_models(returns: pd.Series) -> dict:
    """
    Fits ARCH(1), GARCH(1,1) and EGARCH(1,1) for comparison.
    Multiplied by 100 to improve optimizer convergence.
    """
    scaled = returns * 100

    res_arch   = arch_model(scaled, vol="ARCH",   p=1, q=0).fit(disp="off")
    res_garch  = arch_model(scaled, vol="GARCH",  p=1, q=1).fit(disp="off")
    res_egarch = arch_model(scaled, vol="EGARCH", p=1, q=1).fit(disp="off")

    models = {
        "ARCH(1)":    res_arch.aic,
        "GARCH(1,1)": res_garch.aic,
        "EGARCH(1,1)": res_egarch.aic,
    }
    best_model = min(models, key=models.get)

    return {
        "ARCH(1)": {
            "AIC": res_arch.aic,
            "BIC": res_arch.bic,
            "LogL": res_arch.loglikelihood,
        },
        "GARCH(1,1)": {
            "AIC": res_garch.aic,
            "BIC": res_garch.bic,
            "LogL": res_garch.loglikelihood,
            "Volatility": (res_garch.conditional_volatility / 100).tolist(),
        },
        "EGARCH(1,1)": {
            "AIC": res_egarch.aic,
            "BIC": res_egarch.bic,
            "LogL": res_egarch.loglikelihood,
        },
        "best_model": best_model,
    }


# =============================================================================
# SECCIÓN 3b: VOLATILIDAD EWMA — RiskMetrics (MÓDULO 3)
# =============================================================================

def compute_ewma_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
    init_window: int = 20,
) -> pd.Series:
    """
    Estima la volatilidad mediante un suavizamiento exponencial (EWMA).

    Recursión:  σ²ₜ = λ · σ²ₜ₋₁ + (1 − λ) · r²ₜ₋₁

    El default λ = 0.94 es el estándar RiskMetrics para datos diarios.
    La varianza inicial σ²₀ se aproxima con la varianza muestral de las
    primeras `init_window` observaciones para acelerar la convergencia.

    Devuelve la desviación estándar condicional σₜ (mismo unit que los
    retornos de entrada). Para anualizar: σₜ · √252.
    """
    if not 0 < lambda_ < 1:
        raise ValueError(f"lambda debe estar en (0, 1); recibido: {lambda_}")

    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return pd.Series(dtype=float)

    n = len(r)
    sigma2 = np.full(n, np.nan)
    w = max(2, min(init_window, n))
    sigma2[0] = float(r.iloc[:w].var(ddof=1))

    arr = r.values
    one_minus = 1.0 - lambda_
    for t in range(1, n):
        sigma2[t] = lambda_ * sigma2[t - 1] + one_minus * (arr[t - 1] ** 2)

    return pd.Series(np.sqrt(sigma2), index=r.index, name=f"EWMA_λ{lambda_:.2f}")


def compute_ewma_comparison(
    returns: pd.Series,
    lambdas: list[float],
) -> dict:
    """
    Aplica EWMA con varias λ y devuelve {label: lista_de_sigmas}.

    label tiene la forma '0.94', '0.90', etc., redondeado a 2 decimales.
    """
    out: dict = {}
    for lam in lambdas:
        try:
            s = compute_ewma_volatility(returns, lambda_=float(lam))
            out[f"{float(lam):.2f}"] = s.tolist()
        except (ValueError, TypeError):
            continue
    return out


def ewma_vs_garch_table() -> list[dict]:
    """
    Tabla comparativa EWMA vs GARCH(1,1) (datos cualitativos del informe).

    Las celdas son strings listos para mostrar en el dashboard.
    """
    return [
        {
            "aspect": "Parámetros estimados",
            "ewma": "0 (λ fijo o calibrado)",
            "garch": "3 (ω, α, β)",
        },
        {
            "aspect": "Varianza incondicional",
            "ewma": "No definida",
            "garch": "σ² = ω / (1 − α − β)",
        },
        {
            "aspect": "Reversión a la media",
            "ewma": "No",
            "garch": "Sí, si α + β < 1",
        },
        {
            "aspect": "Costo computacional",
            "ewma": "Mínimo (recursión cerrada)",
            "garch": "Optimización por máxima verosimilitud",
        },
        {
            "aspect": "Captura asimetría",
            "ewma": "No",
            "garch": "Sólo en variantes (EGARCH, GJR)",
        },
        {
            "aspect": "Interpretación",
            "ewma": "Decay exponencial constante",
            "garch": "Estructura paramétrica completa",
        },
    ]


# =============================================================================
# SECCIÓN 4: CAPM Y RIESGO SISTEMÁTICO (MÓDULO 4)
# =============================================================================

def calculate_capm(
    returns: pd.Series,
    bench_returns: pd.Series,
    rf_rate: float = 0.04,
) -> dict:
    """
    rf_rate is annualized; converted to daily for CAPM.
    """
    rf_daily = (1 + rf_rate) ** (1 / 252) - 1
    slope, intercept, r_value, _, _ = stats.linregress(bench_returns, returns)
    expected_return = rf_daily + slope * (bench_returns.mean() - rf_daily)
    return {
        "Beta": slope,
        "Alpha": intercept,
        "R_Squared": r_value ** 2,
        "Expected_Return_Daily": expected_return,
        "Expected_Return_Annual": (1 + expected_return) ** 252 - 1,
        "Classification": (
            "Agresivo" if slope > 1.2 else ("Defensivo" if slope < 0.8 else "Neutro")
        ),
    }


# =============================================================================
# SECCIÓN 5: VALOR EN RIESGO — VaR (MÓDULO 5)
# =============================================================================

def calculate_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
) -> dict:
    """
    Computes Historical, Parametric (Normal) and Monte Carlo VaR/CVaR.
    Returns losses as positive values (conventional VaR sign).
    """
    percentile = (1 - confidence) * 100

    # 1. Historical
    var_hist  = np.percentile(returns, percentile)
    cvar_hist = returns[returns <= var_hist].mean()

    # 2. Parametric (Normal)
    mu, sigma = returns.mean(), returns.std()
    z = stats.norm.ppf(1 - confidence)
    var_param  = mu + sigma * z
    cvar_param = mu - sigma * stats.norm.pdf(z) / (1 - confidence)

    # 3. Monte Carlo
    sims   = np.random.normal(mu, sigma, n_simulations)
    var_mc  = np.percentile(sims, percentile)
    cvar_mc = sims[sims <= var_mc].mean()

    return {
        "Historico":   {"VaR": abs(var_hist),  "CVaR": abs(cvar_hist)},
        "Parametrico": {"VaR": abs(var_param),  "CVaR": abs(cvar_param)},
        "Montecarlo":  {"VaR": abs(var_mc),     "CVaR": abs(cvar_mc)},
        "confidence":     confidence,
        "n_simulations":  n_simulations,
    }


def kupiec_test(returns: pd.Series, var_value: float, confidence: float = 0.95) -> dict:
    """
    Kupiec Proportional Failure Rate (POF) test for VaR backtesting.
    var_value: positive VaR magnitude (loss expressed as positive number).
    LR statistic follows chi-squared(1) under H0 (model is correctly specified).
    """
    T = len(returns)
    p = 1.0 - confidence          # expected daily exception probability
    N = int((returns < -var_value).sum())
    p_hat = N / T if T > 0 else 0.0

    if 0 < p_hat < 1:
        lr_stat = float(-2 * (
            N * np.log(p / p_hat) + (T - N) * np.log((1 - p) / (1 - p_hat))
        ))
        p_value = float(stats.chi2.sf(lr_stat, df=1))
    else:
        lr_stat = None
        p_value = None

    return {
        "T":                   T,
        "N_exceptions":        N,
        "expected_exceptions": round(p * T, 2),
        "exception_rate_pct":  round(p_hat * 100, 4),
        "expected_rate_pct":   round(p * 100, 4),
        "LR_stat":             round(lr_stat, 6) if lr_stat is not None else None,
        "p_value":             round(p_value, 6) if p_value is not None else None,
        "passed":              (p_value > 0.05) if p_value is not None else None,
    }


# =============================================================================
# SECCIÓN 6: OPTIMIZACIÓN DE MARKOWITZ (MÓDULO 6)
# =============================================================================

def optimize_portfolio(
    returns_df: pd.DataFrame,
    n_simulations: int = 10_000,
    rf_rate: float = 0.04,
) -> dict:
    """Random portfolio simulation to trace the efficient frontier."""
    tickers    = returns_df.columns.tolist()
    n_assets   = len(tickers)
    mean_ret   = returns_df.mean()
    cov        = returns_df.cov()

    ret_arr  = np.zeros(n_simulations)
    vol_arr  = np.zeros(n_simulations)
    sharpe_arr = np.zeros(n_simulations)
    weights_rec = []

    for i in range(n_simulations):
        w = np.random.random(n_assets)
        w /= w.sum()
        weights_rec.append(w)
        port_ret = np.sum(mean_ret * w) * 252
        port_vol = np.sqrt(w @ (cov * 252) @ w)
        ret_arr[i]    = port_ret
        vol_arr[i]    = port_vol
        sharpe_arr[i] = (port_ret - rf_rate) / port_vol if port_vol > 0 else 0

    ms_idx  = int(np.argmax(sharpe_arr))
    mv_idx  = int(np.argmin(vol_arr))

    return {
        "Max_Sharpe": {
            "Return":     ret_arr[ms_idx],
            "Volatility": vol_arr[ms_idx],
            "Sharpe":     sharpe_arr[ms_idx],
            "Weights":    dict(zip(tickers, weights_rec[ms_idx].tolist())),
        },
        "Min_Volatility": {
            "Return":     ret_arr[mv_idx],
            "Volatility": vol_arr[mv_idx],
            "Sharpe":     sharpe_arr[mv_idx],
            "Weights":    dict(zip(tickers, weights_rec[mv_idx].tolist())),
        },
        "Correlation": returns_df.corr().to_dict(),
    }


# =============================================================================
# SECCIÓN 6b: MARKOWITZ COMO QP EXPLÍCITO — cvxpy (MÓDULO 6)
# =============================================================================
#
# Formulación canónica de Markowitz como programación cuadrática:
#
#     minimizar    wᵀ Σ w                  (varianza anualizada)
#     sujeto a     Σᵢ wᵢ  = 1              (los pesos suman 1)
#                  μᵀ w   = μ*             (rendimiento objetivo — opcional)
#                  wᵢ    ≥ 0  ∀ i          (no negatividad — opcional)
#
# Se resuelve con cvxpy: cp.Minimize(cp.quad_form(w, Σ)) sujeto a las
# restricciones lineales. La spec del Proyecto Integrador exige
# explícitamente esta formulación y la comparación de la versión con y
# sin la restricción de no negatividad.


def solve_markowitz_qp(
    returns_df: pd.DataFrame,
    target_return: Optional[float] = None,
    allow_short: bool = False,
    rf_rate: float = 0.04,
) -> dict:
    """
    Resuelve un problema cuadrático individual de Markowitz.

    - target_return = None y allow_short=False → mínima varianza global con
      pesos no negativos.
    - target_return = None y allow_short=True  → mínima varianza global con
      pesos libres (permite ventas en corto).
    - target_return = X                        → mínima varianza alcanzando
      el rendimiento anual objetivo X.
    """
    import cvxpy as cp

    tickers = returns_df.columns.tolist()
    n = len(tickers)
    mu_annual = (returns_df.mean() * 252).values
    Sigma_annual = (returns_df.cov() * 252).values

    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1]
    if target_return is not None:
        constraints.append(mu_annual @ w == float(target_return))
    if not allow_short:
        constraints.append(w >= 0)

    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma_annual)))
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve()
    except Exception as exc:
        return {"feasible": False, "error": f"cvxpy: {exc}"}

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"feasible": False, "error": f"status={prob.status}"}

    w_opt = np.asarray(w.value).flatten()
    # Limpiar pequeños valores numéricos (~1e-12) cuando hay no-negatividad
    if not allow_short:
        w_opt = np.clip(w_opt, 0.0, 1.0)
        s = w_opt.sum()
        if s > 0:
            w_opt = w_opt / s

    port_ret = float(np.dot(w_opt, mu_annual))
    port_var = float(w_opt @ Sigma_annual @ w_opt)
    port_vol = math.sqrt(max(port_var, 0.0))
    sharpe = (port_ret - rf_rate) / port_vol if port_vol > 1e-12 else 0.0

    return {
        "feasible": True,
        "allow_short": allow_short,
        "target_return": target_return,
        "Return": port_ret,
        "Volatility": port_vol,
        "Sharpe": sharpe,
        "Weights": dict(zip(tickers, w_opt.tolist())),
    }


def compute_efficient_frontier_qp(
    returns_df: pd.DataFrame,
    allow_short: bool = False,
    n_points: int = 50,
    rf_rate: float = 0.04,
) -> dict:
    """
    Traza la frontera eficiente resolviendo el QP en `n_points` valores de
    rendimiento objetivo distribuidos uniformemente entre la mínima varianza
    global y el activo de máximo rendimiento.

    Retorna además:
      - Min_Variance: portafolio de mínima varianza global.
      - Max_Sharpe:   portafolio con mayor ratio de Sharpe sobre la frontera.
      - Correlation:  matriz de correlación entre activos (para el heatmap).
    """
    mv = solve_markowitz_qp(
        returns_df, target_return=None, allow_short=allow_short, rf_rate=rf_rate
    )
    if not mv.get("feasible"):
        return {
            "feasible": False,
            "error": mv.get("error", "no se pudo resolver min variance"),
            "allow_short": allow_short,
        }

    mu_annual = (returns_df.mean() * 252).values
    min_ret = float(mv["Return"])
    if allow_short:
        # Con ventas en corto el rendimiento alcanzable no está acotado por
        # max(μᵢ); ampliamos el rango para visualizar la rama superior.
        max_ret = float(np.max(mu_annual)) * 2.0
    else:
        max_ret = float(np.max(mu_annual))

    n_points = max(10, int(n_points))
    targets = np.linspace(min_ret, max_ret, n_points)

    frontier: list[dict] = []
    for t in targets:
        sol = solve_markowitz_qp(
            returns_df,
            target_return=float(t),
            allow_short=allow_short,
            rf_rate=rf_rate,
        )
        if sol.get("feasible"):
            frontier.append(
                {
                    "Volatility": sol["Volatility"],
                    "Return":     sol["Return"],
                    "Sharpe":     sol["Sharpe"],
                }
            )

    if frontier:
        ms_idx = max(range(len(frontier)), key=lambda i: frontier[i]["Sharpe"])
        ms_target = frontier[ms_idx]["Return"]
        max_sharpe = solve_markowitz_qp(
            returns_df,
            target_return=ms_target,
            allow_short=allow_short,
            rf_rate=rf_rate,
        )
    else:
        max_sharpe = mv

    return {
        "feasible":     True,
        "allow_short":  allow_short,
        "n_points":     len(frontier),
        "frontier":     frontier,
        "Min_Variance": mv,
        "Max_Sharpe":   max_sharpe,
        "Correlation":  returns_df.corr().to_dict(),
    }


def compare_markowitz_with_without_short(
    returns_df: pd.DataFrame,
    rf_rate: float = 0.04,
    n_points: int = 50,
) -> dict:
    """
    Resuelve Markowitz en dos versiones (con y sin no-negatividad) y reporta
    el costo de imponer la restricción. La spec exige esta comparación.
    """
    res_short = compute_efficient_frontier_qp(
        returns_df, allow_short=True, n_points=n_points, rf_rate=rf_rate
    )
    res_no_short = compute_efficient_frontier_qp(
        returns_df, allow_short=False, n_points=n_points, rf_rate=rf_rate
    )

    def _count_neg(weights: dict) -> int:
        return sum(1 for v in weights.values() if v < -1e-6)

    def _count_zero(weights: dict) -> int:
        return sum(1 for v in weights.values() if abs(v) < 1e-6)

    summary: dict = {}
    if res_short.get("feasible") and res_no_short.get("feasible"):
        ms_s  = res_short["Max_Sharpe"]
        ms_ns = res_no_short["Max_Sharpe"]
        mv_s  = res_short["Min_Variance"]
        mv_ns = res_no_short["Min_Variance"]
        summary = {
            "max_sharpe_with_short": {
                "Sharpe":             ms_s["Sharpe"],
                "Volatility":         ms_s["Volatility"],
                "Return":             ms_s["Return"],
                "n_negative_weights": _count_neg(ms_s["Weights"]),
                "Weights":            ms_s["Weights"],
            },
            "max_sharpe_no_short": {
                "Sharpe":         ms_ns["Sharpe"],
                "Volatility":     ms_ns["Volatility"],
                "Return":         ms_ns["Return"],
                "n_zero_weights": _count_zero(ms_ns["Weights"]),
                "Weights":        ms_ns["Weights"],
            },
            "min_variance_with_short": {
                "Volatility":         mv_s["Volatility"],
                "Return":             mv_s["Return"],
                "n_negative_weights": _count_neg(mv_s["Weights"]),
            },
            "min_variance_no_short": {
                "Volatility":     mv_ns["Volatility"],
                "Return":         mv_ns["Return"],
                "n_zero_weights": _count_zero(mv_ns["Weights"]),
            },
            "cost_of_no_short": {
                "extra_volatility_max_sharpe":  ms_ns["Volatility"] - ms_s["Volatility"],
                "lower_sharpe_max_sharpe":      ms_s["Sharpe"]      - ms_ns["Sharpe"],
                "extra_volatility_min_var":     mv_ns["Volatility"] - mv_s["Volatility"],
            },
        }

    return {
        "with_short":         res_short,
        "no_short":           res_no_short,
        "comparison_summary": summary,
    }


def optimize_portfolio_target_return(
    returns_df: pd.DataFrame,
    target_return: float,
    rf_rate: float = 0.04,
) -> dict:
    """
    Find minimum-volatility portfolio that achieves target_return (annualized).
    Uses SLSQP constrained optimization.
    """
    from scipy.optimize import minimize

    tickers  = returns_df.columns.tolist()
    n        = len(tickers)
    mean_ann = returns_df.mean() * 252
    cov_ann  = returns_df.cov()  * 252

    def _vol(w):
        return float(np.sqrt(w @ cov_ann.values @ w))

    constraints = [
        {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
        {"type": "eq", "fun": lambda w: float(np.dot(w, mean_ann.values)) - target_return},
    ]
    bounds = [(0.0, 1.0)] * n
    w0     = np.ones(n) / n

    res = minimize(
        _vol, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not res.success:
        return {"feasible": False, "error": "No existe portafolio factible con ese rendimiento objetivo. Prueba un valor dentro del rango de retornos individuales."}

    w       = np.clip(res.x, 0, 1)
    w      /= w.sum()
    port_ret = float(np.dot(w, mean_ann.values))
    port_vol = _vol(w)
    sharpe   = (port_ret - rf_rate) / port_vol if port_vol > 0 else 0.0

    return {
        "feasible":      True,
        "target_return": target_return,
        "Return":        port_ret,
        "Volatility":    port_vol,
        "Sharpe":        sharpe,
        "Weights":       dict(zip(tickers, w.tolist())),
    }


# =============================================================================
# SECCIÓN 7: SEÑALES Y ALERTAS (MÓDULO 7)
# =============================================================================

def generate_signals(data_dict: dict) -> list:
    """
    Generates Buy/Sell alerts based on technical indicators.
    Returns a list of signal dicts with id, msg, type and explanation.
    """
    signals = []

    rsi      = data_dict.get("RSI", 50)
    macd_h   = data_dict.get("MACD_Hist", 0)
    macd_p   = data_dict.get("MACD_Hist_Prev", 0)
    close    = data_dict.get("Close", 0)
    bb_up    = data_dict.get("BB_Upper", 0)
    bb_low   = data_dict.get("BB_Lower", 0)

    if rsi > 70:
        signals.append({
            "id": "RSI",
            "msg": "SOBRECOMPRA — RSI supera 70",
            "type": "Sell",
            "explanation": (
                f"El índice RSI ({rsi:.1f}) está por encima de 70, lo que indica que el activo "
                "ha subido muy rápido en los últimos días. Esto no garantiza una caída, pero "
                "sugiere que el precio puede estar sobreextendido y podría hacer una pausa o retroceso."
            ),
        })
    elif rsi < 30:
        signals.append({
            "id": "RSI",
            "msg": "SOBREVENTA — RSI por debajo de 30",
            "type": "Buy",
            "explanation": (
                f"El RSI ({rsi:.1f}) está por debajo de 30, señalando que el activo ha caído "
                "aceleradamente. El mercado puede estar siendo excesivamente pesimista, "
                "lo que históricamente ha coincidido con rebotes o recuperaciones de precio."
            ),
        })

    if macd_h > 0 and macd_p <= 0:
        signals.append({
            "id": "MACD",
            "msg": "CRUCE ALCISTA — Histograma MACD positivo",
            "type": "Buy",
            "explanation": (
                "El histograma MACD acaba de cruzar hacia territorio positivo. "
                "Esto indica que el momentum de corto plazo supera al de largo plazo, "
                "una señal técnica que los operadores asocian con arranques alcistas."
            ),
        })
    elif macd_h < 0 and macd_p >= 0:
        signals.append({
            "id": "MACD",
            "msg": "CRUCE BAJISTA — Histograma MACD negativo",
            "type": "Sell",
            "explanation": (
                "El histograma MACD acaba de cruzar hacia territorio negativo. "
                "El momentum de corto plazo ha cedido frente al de largo plazo, "
                "lo que puede anticipar una corrección o desaceleración del precio."
            ),
        })

    if close >= bb_up:
        signals.append({
            "id": "BB",
            "msg": "PRECIO EN BANDA SUPERIOR de Bollinger",
            "type": "Sell",
            "explanation": (
                "El precio ha llegado o superado la banda superior de Bollinger, que representa "
                "aproximadamente dos desviaciones estándar sobre el precio promedio de los últimos "
                "20 días. Estadísticamente, el precio tiende a revertir desde estos extremos. "
                "Esto puede ocurrir porque el activo ha recibido noticias muy positivas o hay "
                "un exceso de optimismo en el mercado."
            ),
        })
    elif close <= bb_low:
        signals.append({
            "id": "BB",
            "msg": "PRECIO EN BANDA INFERIOR de Bollinger",
            "type": "Buy",
            "explanation": (
                "El precio tocó o perforó la banda inferior de Bollinger, indicando que cotiza "
                "muy por debajo de su rango habitual. Este tipo de lectura extrema puede señalar "
                "una oportunidad de compra si los fundamentos del activo no han cambiado, "
                "aunque también puede indicar una tendencia bajista sostenida."
            ),
        })

    return signals
