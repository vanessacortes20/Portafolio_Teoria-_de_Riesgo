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


# =============================================================================
# SECCIÓN 6: OPTIMIZACIÓN DE MARKOWITZ (MÓDULO 6)
# =============================================================================

def optimize_portfolio(
    returns_df: pd.DataFrame,
    n_simulations: int = 5_000,
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
        sharpe_arr[i] = port_ret / port_vol if port_vol > 0 else 0

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
