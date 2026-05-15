import pandas as pd
import numpy as np
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


def calculate_ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> dict:
    """
    Volatilidad EWMA (RiskMetrics): σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}.

    lambda_ = factor de decaimiento (0 < λ < 1). Default 0.94 (RiskMetrics estándar).
    Devuelve la serie de desviación estándar diaria, último valor, y comparación
    contra la volatilidad muestral rodante.
    """
    if not 0 < lambda_ < 1:
        raise ValueError(f"lambda_ debe estar en (0,1); recibido: {lambda_}")

    r = returns.dropna()
    # pandas usa alpha = 1 - lambda
    var_ewma = r.pow(2).ewm(alpha=1 - lambda_, adjust=False).mean()
    sigma_ewma = var_ewma.pow(0.5)

    rolling_30 = r.rolling(window=30).std().dropna()
    rolling_30_avg = float(rolling_30.mean()) if len(rolling_30) else None

    return {
        "lambda":          lambda_,
        "ewma_volatility": sigma_ewma.tolist(),
        "ewma_last_value": float(sigma_ewma.iloc[-1]) if len(sigma_ewma) else None,
        "ewma_mean":       float(sigma_ewma.mean()) if len(sigma_ewma) else None,
        "rolling_30d_avg": rolling_30_avg,
        "n_obs":           int(len(r)),
    }


def arch_lm_test(std_residuals: list[float] | pd.Series, nlags: int = 5) -> dict:
    """
    Test ARCH-LM (Engle 1982) sobre residuos estandarizados.

    Detecta heterocedasticidad condicional remanente. Bajo H0 (no hay efecto
    ARCH), el estadístico LM ~ χ²(nlags).
    p > 0.05 → el modelo capturó adecuadamente la volatilidad condicional.
    p ≤ 0.05 → quedan efectos ARCH no capturados; conviene un orden mayor.
    """
    from statsmodels.stats.diagnostic import het_arch
    s = pd.Series(std_residuals).dropna()
    if len(s) < 2 * nlags + 5:
        return {"lm_stat": None, "lm_pvalue": None, "passed": None,
                "interpretation": "muestra insuficiente para ARCH-LM"}
    try:
        lm_stat, lm_p, _f_stat, _f_p = het_arch(s, nlags=nlags)
        passed = bool(lm_p > 0.05)
        interp = (
            "no se detectan efectos ARCH residuales — el modelo captura bien la heterocedasticidad"
            if passed else
            "se detectan efectos ARCH residuales — el modelo deja heterocedasticidad sin capturar"
        )
        return {
            "lm_stat":        round(float(lm_stat), 6),
            "lm_pvalue":      round(float(lm_p), 6),
            "nlags":          nlags,
            "passed":         passed,
            "interpretation": interp,
        }
    except Exception as exc:  # pragma: no cover
        return {"lm_stat": None, "lm_pvalue": None, "passed": None,
                "interpretation": f"error en ARCH-LM: {exc}"}


def compare_ewma_vs_garch(
    ewma_last: float | None,
    garch_last: float | None,
) -> dict:
    """Comparación interpretativa entre el último valor EWMA y GARCH(1,1)."""
    if ewma_last is None or garch_last is None:
        return {"diff_pct": None, "interpretation": "datos insuficientes"}
    diff = ewma_last - garch_last
    diff_pct = round((diff / garch_last) * 100, 3) if garch_last else None
    if abs(diff_pct or 0) < 5:
        msg = "EWMA y GARCH(1,1) coinciden — el régimen de volatilidad actual es estable"
    elif diff_pct and diff_pct > 0:
        msg = f"EWMA estima volatilidad {abs(diff_pct):.1f}% mayor que GARCH — choque reciente aún pesa en EWMA"
    else:
        msg = f"EWMA estima volatilidad {abs(diff_pct):.1f}% menor que GARCH — el shock se está disipando"
    return {
        "ewma_last":      round(ewma_last, 6),
        "garch_last":     round(garch_last, 6),
        "diff_pct":       diff_pct,
        "interpretation": msg,
    }


# =============================================================================
# SECCIÓN 4: CAPM Y RIESGO SISTEMÁTICO (MÓDULO 4)
# =============================================================================

def calculate_capm(
    returns: pd.Series,
    bench_returns: pd.Series,
    rf_rate: float = 0.04,
) -> dict:
    """
    CAPM por regresión OLS de retornos del activo contra retornos del benchmark.

    rf_rate es anualizado (decimal). Se convierte a diario internamente.

    Devuelve:
    - Beta, Alpha, R_Squared (regresión MCO)
    - Expected_Return_Daily / _Annual según CAPM: E[R] = Rf + β·(E[Rm] - Rf)
    - Classification: agresivo (β>1.05), defensivo (β<0.95), neutro (β≈1)
    - Variance_Decomposition: sistemática (β²·σ²_m) vs idiosincrática (σ²_ε)
    """
    rf_daily = (1 + rf_rate) ** (1 / 252) - 1
    slope, intercept, r_value, _, std_err = stats.linregress(bench_returns, returns)
    market_mean_daily = float(bench_returns.mean())
    market_premium    = market_mean_daily - rf_daily
    expected_return   = rf_daily + slope * market_premium

    # ── Descomposición de varianza ──────────────────────────────────────────
    var_market = float(bench_returns.var())
    var_total  = float(returns.var())
    var_systematic    = float((slope ** 2) * var_market)
    var_idiosyncratic = float(max(var_total - var_systematic, 0.0))
    sys_share = float(var_systematic / var_total) if var_total > 0 else None

    # Clasificación con tolerancia ±5% alrededor de β=1
    if slope > 1.05:
        classification = "Agresivo"
        class_note = (
            f"β={slope:.3f} > 1.05 → amplifica los movimientos del mercado. "
            "Riesgo sistemático mayor que el promedio."
        )
    elif slope < 0.95:
        classification = "Defensivo"
        class_note = (
            f"β={slope:.3f} < 0.95 → menos sensible al mercado. "
            "Útil como diversificador en caídas sistémicas."
        )
    else:
        classification = "Neutro"
        class_note = f"β≈1 ({slope:.3f}) → se mueve aproximadamente como el mercado."

    return {
        "Beta": slope,
        "Alpha": intercept,
        "R_Squared": r_value ** 2,
        "Beta_StdErr": float(std_err) if std_err is not None else None,
        "Rf_Annual_Used":  rf_rate,
        "Rf_Daily_Used":   rf_daily,
        "Market_Mean_Daily":    market_mean_daily,
        "Market_Premium_Daily": market_premium,
        "Expected_Return_Daily": expected_return,
        "Expected_Return_Annual": (1 + expected_return) ** 252 - 1,
        "Classification": classification,
        "Classification_Note": class_note,
        "Variance_Decomposition": {
            "var_total":         var_total,
            "var_systematic":    var_systematic,
            "var_idiosyncratic": var_idiosyncratic,
            "systematic_share":  sys_share,
            "interpretation":    (
                f"{(sys_share or 0)*100:.1f}% de la varianza viene del mercado (riesgo sistemático no diversificable). "
                f"El restante {100 - (sys_share or 0)*100:.1f}% es riesgo idiosincrático: se reduce al combinar con activos descorrelacionados."
            ),
        },
    }


# =============================================================================
# SECCIÓN 5: VALOR EN RIESGO — VaR (MÓDULO 5)
# =============================================================================

def calculate_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Computes Historical, Parametric (Normal) and Monte Carlo VaR/CVaR.

    Returns losses as positive values (conventional VaR sign).

    seed: semilla NumPy para reproducibilidad del Monte Carlo. Default 42.
          La distribución asumida es Normal(μ, σ) con μ y σ empíricos de
          los retornos del portafolio.
    """
    percentile = (1 - confidence) * 100

    # 1. Historical (no paramétrico — percentil empírico)
    var_hist  = np.percentile(returns, percentile)
    cvar_hist = returns[returns <= var_hist].mean()

    # 2. Parametric (asume Normal)
    mu, sigma = returns.mean(), returns.std()
    z = stats.norm.ppf(1 - confidence)
    var_param  = mu + sigma * z
    cvar_param = mu - sigma * stats.norm.pdf(z) / (1 - confidence)

    # 3. Monte Carlo (Normal(μ, σ) con semilla fija para reproducibilidad)
    rng    = np.random.default_rng(seed)
    sims   = rng.normal(mu, sigma, n_simulations)
    var_mc  = np.percentile(sims, percentile)
    cvar_mc = sims[sims <= var_mc].mean()

    return {
        "Historico":   {"VaR": abs(var_hist),  "CVaR": abs(cvar_hist)},
        "Parametrico": {"VaR": abs(var_param),  "CVaR": abs(cvar_param)},
        "Montecarlo":  {"VaR": abs(var_mc),     "CVaR": abs(cvar_mc),
                        "distribution": "Normal(mu_emp, sigma_emp)",
                        "seed": seed},
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

    # Interpretación textual: ¿el modelo subestima o sobreestima el riesgo?
    if lr_stat is None:
        verdict = "no determinable"
        interp = (f"No hay excepciones suficientes para evaluar (N={N} de T={T}). "
                  f"El test requiere 0 < p̂ < 1.")
    else:
        passes = bool(p_value > 0.05)
        if passes:
            verdict = "modelo correcto"
            interp = (f"Tasa observada p̂={p_hat*100:.2f}% vs esperada {p*100:.2f}%. "
                      f"LR_POF={lr_stat:.3f} ≤ 3.84 (chi²(1) al 95%); p={p_value:.4g}. "
                      f"No se rechaza H0 → la frecuencia de excedencias es coherente con el VaR declarado.")
        elif p_hat > p:
            verdict = "subestima el riesgo"
            interp = (f"Tasa observada p̂={p_hat*100:.2f}% > esperada {p*100:.2f}%. "
                      f"LR_POF={lr_stat:.3f} > 3.84 → se rechaza H0. "
                      f"El modelo SUBESTIMA el riesgo: ocurren más excedencias de las que el VaR predice.")
        else:
            verdict = "sobreestima el riesgo"
            interp = (f"Tasa observada p̂={p_hat*100:.2f}% < esperada {p*100:.2f}%. "
                      f"LR_POF={lr_stat:.3f} > 3.84 → se rechaza H0. "
                      f"El modelo SOBREESTIMA el riesgo: el VaR es demasiado conservador.")

    return {
        "T":                   T,
        "N_exceptions":        N,
        "expected_exceptions": round(p * T, 2),
        "exception_rate_pct":  round(p_hat * 100, 4),
        "expected_rate_pct":   round(p * 100, 4),
        "LR_stat":             round(lr_stat, 6) if lr_stat is not None else None,
        "p_value":             round(p_value, 6) if p_value is not None else None,
        "passed":              (p_value > 0.05) if p_value is not None else None,
        "verdict":             verdict,
        "interpretation":      interp,
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


def optimize_portfolio_qp(
    returns_df: pd.DataFrame,
    allow_short: bool = False,
    rf_rate: float = 0.04,
) -> dict:
    """
    Optimización Markowitz por programación cuadrática (QP) explícita usando SLSQP.

    Resuelve dos problemas con las mismas restricciones excepto las cotas de pesos:
      • Mínima varianza global:   min wᵀΣw      s.t. Σwᵢ = 1
      • Máximo Sharpe:            max (wᵀμ - rf)/√(wᵀΣw)   s.t. Σwᵢ = 1

    Si allow_short=False (long-only) los pesos están en [0, 1].
    Si allow_short=True (short-selling permitido) los pesos están en [-1, 1].

    Devuelve los pesos, retorno, volatilidad y Sharpe de cada portafolio óptimo.
    """
    from scipy.optimize import minimize

    tickers  = returns_df.columns.tolist()
    n        = len(tickers)
    mean_ann = returns_df.mean() * 252
    cov_ann  = returns_df.cov()  * 252

    bounds = [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n
    cons   = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
    w0     = np.ones(n) / n

    def _portfolio_vol(w):
        return float(np.sqrt(w @ cov_ann.values @ w))

    def _neg_sharpe(w):
        port_ret = float(np.dot(w, mean_ann.values))
        port_vol = _portfolio_vol(w)
        if port_vol <= 0:
            return 1e10
        return -(port_ret - rf_rate) / port_vol

    # ── Mínima varianza ────────────────────────────────────────────────────
    res_min = minimize(_portfolio_vol, w0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-10})
    w_min   = res_min.x if res_min.success else w0
    ret_min = float(np.dot(w_min, mean_ann.values))
    vol_min = _portfolio_vol(w_min)
    shp_min = (ret_min - rf_rate) / vol_min if vol_min > 0 else 0.0

    # ── Máximo Sharpe ──────────────────────────────────────────────────────
    res_max = minimize(_neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-10})
    w_max   = res_max.x if res_max.success else w0
    ret_max = float(np.dot(w_max, mean_ann.values))
    vol_max = _portfolio_vol(w_max)
    shp_max = (ret_max - rf_rate) / vol_max if vol_max > 0 else 0.0

    def _pack(w, ret, vol, shp, success):
        return {
            "Weights":    dict(zip(tickers, [float(x) for x in w])),
            "Return":     ret,
            "Volatility": vol,
            "Sharpe":     shp,
            "converged":  bool(success),
        }

    return {
        "allow_short":   allow_short,
        "min_variance":  _pack(w_min, ret_min, vol_min, shp_min, res_min.success),
        "max_sharpe":    _pack(w_max, ret_max, vol_max, shp_max, res_max.success),
    }


def efficient_frontier_curve(
    returns_df: pd.DataFrame,
    allow_short: bool = False,
    n_points: int = 30,
) -> dict:
    """
    Genera la frontera eficiente paramétrica resolviendo, para cada μ* del grid,
    el QP `min wᵀΣw  s.t. wᵀμ = μ*, Σwᵢ = 1, [wᵢ ≥ 0 si long-only]`.

    Devuelve listas paralelas `target_returns`, `min_volatility`, y la composición
    de pesos para cada punto. Ideal para que el frontend dibuje la curva resaltada
    sobre el conjunto factible del Monte Carlo.
    """
    from scipy.optimize import minimize

    tickers  = returns_df.columns.tolist()
    n        = len(tickers)
    mean_ann = returns_df.mean() * 252
    cov_ann  = returns_df.cov()  * 252
    bounds   = [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n

    def _vol(w):
        return float(np.sqrt(w @ cov_ann.values @ w))

    # Rango de retornos objetivo: desde el mínimo individual al máximo individual
    mu_min = float(mean_ann.min())
    mu_max = float(mean_ann.max())
    targets = np.linspace(mu_min, mu_max, n_points)

    rets, vols, weights_list, conv = [], [], [], []
    w0 = np.ones(n) / n
    for mu_star in targets:
        cons = [
            {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
            {"type": "eq", "fun": lambda w, mu=mu_star: float(np.dot(w, mean_ann.values)) - mu},
        ]
        try:
            res = minimize(_vol, w0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-9})
            if res.success:
                rets.append(float(np.dot(res.x, mean_ann.values)))
                vols.append(_vol(res.x))
                weights_list.append({t: float(w) for t, w in zip(tickers, res.x)})
                conv.append(True)
            else:
                rets.append(float(mu_star)); vols.append(None); weights_list.append({}); conv.append(False)
        except Exception:
            rets.append(float(mu_star)); vols.append(None); weights_list.append({}); conv.append(False)

    return {
        "allow_short":    allow_short,
        "n_points":       n_points,
        "target_returns": [round(r, 6) for r in rets],
        "min_volatility": [round(v, 6) if v is not None else None for v in vols],
        "weights":        weights_list,
        "converged":      conv,
        "n_converged":    int(sum(conv)),
    }


def compare_qp_long_only_vs_short(
    returns_df: pd.DataFrame,
    rf_rate: float = 0.04,
) -> dict:
    """
    Resuelve QP en las dos versiones (con y sin no-negatividad) y devuelve
    comparación interpretativa con costo cuantificado de la restricción.
    """
    long_only  = optimize_portfolio_qp(returns_df, allow_short=False, rf_rate=rf_rate)
    with_short = optimize_portfolio_qp(returns_df, allow_short=True,  rf_rate=rf_rate)

    # Activos con peso ~0 en long-only (esquina del conjunto factible)
    zero_threshold = 1e-3
    zero_assets = [t for t, w in long_only["max_sharpe"]["Weights"].items()
                   if abs(w) < zero_threshold]
    short_assets = [t for t, w in with_short["max_sharpe"]["Weights"].items()
                    if w < -zero_threshold]

    # Costo cuantificado de imponer no-negatividad sobre el max-Sharpe
    delta_sharpe = round(with_short["max_sharpe"]["Sharpe"]   - long_only["max_sharpe"]["Sharpe"], 6)
    delta_return = round(with_short["max_sharpe"]["Return"]   - long_only["max_sharpe"]["Return"], 6)
    delta_vol    = round(with_short["max_sharpe"]["Volatility"] - long_only["max_sharpe"]["Volatility"], 6)
    # Mismo análisis sobre min-variance
    delta_min_vol = round(with_short["min_variance"]["Volatility"] - long_only["min_variance"]["Volatility"], 6)

    if delta_sharpe > 0.05:
        msg = ("Permitir short-selling mejora notablemente el Sharpe — el modelo aprovecha "
               "ventas en corto para reducir volatilidad. Trade-off: mayor complejidad operativa.")
    elif delta_sharpe > 0.0:
        msg = ("Permitir short-selling mejora marginalmente el Sharpe. El portafolio long-only "
               "ya está cerca del óptimo no restringido.")
    else:
        msg = ("Long-only iguala o supera al portafolio con short — la restricción de no-negatividad "
               "no es vinculante en este conjunto de activos.")

    # Tabla de composición porcentual (pesos en %)
    tickers = list(long_only["max_sharpe"]["Weights"].keys())
    composition_table = []
    for t in tickers:
        composition_table.append({
            "ticker":          t,
            "min_var_long":    round(long_only["min_variance"]["Weights"].get(t, 0) * 100, 3),
            "max_sharpe_long": round(long_only["max_sharpe"]["Weights"].get(t, 0) * 100, 3),
            "min_var_short":   round(with_short["min_variance"]["Weights"].get(t, 0) * 100, 3),
            "max_sharpe_short": round(with_short["max_sharpe"]["Weights"].get(t, 0) * 100, 3),
            "is_zero_in_long_only": t in zero_assets,
            "is_short_when_allowed": t in short_assets,
        })

    return {
        "long_only":            long_only,
        "with_short":           with_short,
        "sharpe_gain_with_short":      delta_sharpe,
        "zero_weight_in_long_only":    zero_assets,
        "short_positions_when_allowed": short_assets,
        "composition_table":           composition_table,
        "restriction_cost": {
            "delta_sharpe_max":   delta_sharpe,    # >0 si short mejora Sharpe
            "delta_return_max":   delta_return,
            "delta_volatility_max": delta_vol,
            "delta_min_variance": delta_min_vol,   # <0 si short reduce mín varianza
            "interpretation": (
                f"Costo de imponer no-negatividad sobre el portafolio máx-Sharpe: "
                f"ΔSharpe = {delta_sharpe:+.4f}, ΔReturn = {delta_return:+.4f}, "
                f"ΔVolatility = {delta_vol:+.4f}. "
                f"Sobre la mín-varianza: ΔVol = {delta_min_vol:+.4f}. "
                f"Activos en cero (long-only): {zero_assets or 'ninguno'}. "
                f"Activos en corto (cuando se permite): {short_assets or 'ninguno'}."
            ),
        },
        "interpretation":       msg,
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
