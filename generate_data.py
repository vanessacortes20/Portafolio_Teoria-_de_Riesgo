#!/usr/bin/env python3
"""
generate_data.py — RiskLab USTA
================================
Descarga datos de Yahoo Finance, ejecuta todos los cálculos analíticos
y guarda los resultados en data.js (misma carpeta que dashboard.html).

Uso básico:
    python generate_data.py

Con parámetros validados por Pydantic:
    python generate_data.py --start-date 2023-01-01 --end-date 2024-12-31
    python generate_data.py --confidence 0.99 --n-simulations 50000
    python generate_data.py --start-date 2022-01-01 --end-date 2024-06-30 --confidence 0.97

Validaciones (idénticas a los endpoints FastAPI):
    · start_date >= 2000-01-01
    · end_date <= hoy (no puede ser futura)
    · end_date > start_date
    · Período mínimo: 30 días
    · confidence: 0.80 – 0.99
    · n_simulations: 1,000 – 100,000
"""

import argparse
import json
import math
import os
import sys
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Forzar UTF-8 en la salida de consola (Windows cp1252 no soporta ✓ ✗ etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from dotenv import load_dotenv

# ── Asegurar que el paquete 'api' sea importable ──────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from api.data import get_historical_data
from api.logic import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_bollinger_bands, calculate_stochastic,
    calculate_returns, get_descriptive_stats, perform_normality_tests,
    fit_garch_models, calculate_capm, calculate_var_cvar,
    optimize_portfolio, generate_signals,
)

# ── Modelo Pydantic: validación de parámetros CLI ─────────────────────────────

class GeneratorConfig(BaseModel):
    """
    Parámetros de generación validados con Pydantic.
    Las mismas reglas que los endpoints FastAPI (DateRangeDep + VaRParamsDep).
    """
    start_date: Optional[str] = Field(
        None, description="Fecha inicio del análisis (YYYY-MM-DD). Mínimo: 2000-01-01."
    )
    end_date: Optional[str] = Field(
        None, description="Fecha fin del análisis (YYYY-MM-DD). No puede ser futura."
    )
    confidence: float = Field(
        0.95, ge=0.80, le=0.99,
        description="Nivel de confianza para VaR/CVaR (entre 0.80 y 0.99)."
    )
    n_simulations: int = Field(
        10_000, ge=1_000, le=100_000,
        description="Iteraciones Monte Carlo (entre 1 000 y 100 000)."
    )
    output: str = Field("data.js", description="Ruta del archivo JS de salida.")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            date_type.fromisoformat(str(v))
        except ValueError:
            raise ValueError(f"Formato de fecha inválido: '{v}'. Use YYYY-MM-DD.")
        return v

    @model_validator(mode="after")
    def validate_date_range(self) -> "GeneratorConfig":
        if self.start_date and self.end_date:
            start = date_type.fromisoformat(self.start_date)
            end   = date_type.fromisoformat(self.end_date)
            today = date_type.today()
            if end > today:
                raise ValueError("end_date no puede ser una fecha futura.")
            if start < date_type(2000, 1, 1):
                raise ValueError("start_date no puede ser anterior al año 2000.")
            if end <= start:
                raise ValueError("end_date debe ser posterior a start_date.")
            if (end - start).days < 30:
                raise ValueError("El período mínimo de análisis es de 30 días.")
        elif bool(self.start_date) != bool(self.end_date):
            raise ValueError(
                "Debes especificar ambas fechas (start_date y end_date) o ninguna."
            )
        return self


# ── Configuración ─────────────────────────────────────────────────────────────
TICKERS   = [t.strip() for t in os.getenv("PORTFOLIO_TICKERS", "NU,AMZN,SONY,XOM,WPM").split(",")]
BENCHMARK = os.getenv("BENCHMARK_TICKER", "^GSPC")
OUT_FILE  = ROOT / "frontend" / "data.js"


# ── Utilidades de serialización ───────────────────────────────────────────────
def clean(v):
    """Convierte tipos numpy y NaN/Inf en valores JSON-seguros."""
    if isinstance(v, dict):
        return {k: clean(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(i) for i in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if hasattr(v, "item"):
        return clean(v.item())
    return v


def df_records(df: pd.DataFrame) -> list:
    """DataFrame → lista de dicts con NaN limpiados."""
    df = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(df.to_json(orient="records"))


# ── Tasa libre de riesgo dinámica ─────────────────────────────────────────────
def get_rf_rate() -> tuple:
    """
    Obtiene la tasa libre de riesgo del T-Bill a 13 semanas (^IRX).
    Retorna (rf_annual, source_label).
    """
    try:
        irx_data = get_historical_data("^IRX")
        if irx_data is not None and len(irx_data) > 0:
            # ^IRX cotiza en porcentaje anualizado (ej. 4.5 = 4.5%)
            rf_pct = float(irx_data["Close"].iloc[-1])
            rf_annual = rf_pct / 100.0
            return rf_annual, f"^IRX ({rf_pct:.2f}%)"
    except Exception:
        pass
    return 0.045, "Default (4.5%)"


# ── Funciones de cómputo ──────────────────────────────────────────────────────
def compute_technical(data: pd.DataFrame) -> list:
    df = data.copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["SMA_20"] = calculate_sma(data, 20)
    df["SMA_50"] = calculate_sma(data, 50)
    df["EMA_20"] = calculate_ema(data)
    df["RSI"]    = calculate_rsi(data)

    ml, sl, hl = calculate_macd(data)
    df["MACD_Line"]   = ml
    df["MACD_Signal"] = sl
    df["MACD_Hist"]   = hl

    bu, bl = calculate_bollinger_bands(data)
    df["BB_Upper"] = bu
    df["BB_Lower"] = bl

    k, d = calculate_stochastic(data)
    df["Stoch_K"] = k
    df["Stoch_D"] = d

    return df_records(df)


def compute_returns(data: pd.DataFrame) -> dict:
    simple, log = calculate_returns(data)

    # Q-Q data (teórico vs empírico)
    sorted_ret = np.sort(simple.values)
    n = len(sorted_ret)
    theoretical_q = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, n))
    empirical_q = np.interp(
        np.linspace(0.01, 0.99, n),
        np.linspace(0, 1, n),
        sorted_ret,
    )

    # Stylized facts
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(simple**2, lags=[10], return_df=True)
        lb_stat  = float(lb_result["lb_stat"].iloc[0])
        lb_pval  = float(lb_result["lb_pvalue"].iloc[0])
        vol_clustering = lb_pval < 0.05
    except Exception:
        lb_stat, lb_pval, vol_clustering = None, None, None

    skew_val = float(simple.skew())
    kurt_val = float(simple.kurtosis())  # excess kurtosis

    return {
        "stats":     clean(get_descriptive_stats(simple)),
        "normality": clean(perform_normality_tests(simple)),
        "plot_data": {
            "Simple_Returns": clean(simple.replace([np.inf, -np.inf], np.nan).fillna(0).tolist()),
            "Log_Returns":    clean(log.replace([np.inf, -np.inf], np.nan).fillna(0).tolist()),
            "Dates":          data["Date"].iloc[1:].dt.strftime("%Y-%m-%d").tolist(),
        },
        "qq_data": {
            "Theoretical": clean(theoretical_q.tolist()),
            "Empirical":   clean(empirical_q.tolist()),
        },
        "stylized_facts": {
            "Skewness":          clean(skew_val),
            "Excess_Kurtosis":   clean(kurt_val),
            "LB_Stat":           clean(lb_stat),
            "LB_Pvalue":         clean(lb_pval),
            "Vol_Clustering":    vol_clustering,
            "Neg_Skew":          skew_val < 0,
            "Fat_Tails":         kurt_val > 1.0,
        },
    }


def compute_volatility(data: pd.DataFrame) -> dict:
    _, log = calculate_returns(data)
    result = clean(fit_garch_models(log))

    # Extraer residuos estandarizados y pronóstico de GARCH(1,1)
    try:
        from arch import arch_model
        scaled = log * 100
        m = arch_model(scaled, vol="GARCH", p=1, q=1)
        res = m.fit(disp="off")
        std_resid = res.std_resid.dropna().tolist()
        # Pronóstico 10 días
        forecast = res.forecast(horizon=10)
        vol_forecast_raw = forecast.variance.iloc[-1].values
        vol_forecast = (np.sqrt(vol_forecast_raw) / 100).tolist()

        # Jarque-Bera sobre residuos
        jb_stat, jb_p = scipy_stats.jarque_bera(std_resid)

        result["Residuals"] = {
            "Std_Residuals": clean(std_resid[-500:]),  # últimos 500 para no sobrecargar
            "JB_Stat":       clean(float(jb_stat)),
            "JB_Pvalue":     clean(float(jb_p)),
            "Normal":        bool(jb_p > 0.05),
        }
        result["Forecast_10d"] = clean(vol_forecast)
    except Exception as e:
        print(f"        ! Residuals/forecast: {e}", flush=True)

    return result


def compute_risk(
    data: pd.DataFrame,
    bench_data: pd.DataFrame,
    rf_annual: float = 0.045,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
) -> dict:
    ret,  _ = calculate_returns(data)
    bret, _ = calculate_returns(bench_data)
    common  = ret.index.intersection(bret.index)

    if len(common) < 10:
        capm = {
            "Beta": 1.0, "Alpha": 0.0, "R_Squared": 0.0,
            "Expected_Return_Daily": 0.0, "Expected_Return_Annual": 0.0,
            "Classification": "Sin Datos Suficientes",
        }
        scatter = {"Asset": [], "Benchmark": [], "Fitted": []}
    else:
        capm = calculate_capm(ret.loc[common], bret.loc[common], rf_annual)
        # Scatter: retornos alineados + línea de regresión
        asset_vals = ret.loc[common].tolist()
        bench_vals = bret.loc[common].tolist()
        x_min, x_max = min(bench_vals), max(bench_vals)
        x_line = [x_min, x_max]
        y_line = [capm["Alpha"] + capm["Beta"] * x for x in x_line]
        scatter = {
            "Asset":     clean(asset_vals),
            "Benchmark": clean(bench_vals),
            "Reg_X":     clean(x_line),
            "Reg_Y":     clean(y_line),
        }

    var_95 = calculate_var_cvar(ret, confidence=confidence, n_simulations=n_simulations)
    var_99 = calculate_var_cvar(ret, confidence=min(confidence + 0.04, 0.99), n_simulations=n_simulations)

    # Anualizaciones: VaR × √252
    sqrt252 = math.sqrt(252)
    def annualize_var(v):
        out = {}
        for method, vals in v.items():
            if not isinstance(vals, dict):
                continue
            out[method] = {
                "VaR":       vals["VaR"],
                "CVaR":      vals["CVaR"],
                "VaR_Ann":   clean(vals["VaR"] * sqrt252) if vals["VaR"] is not None else None,
                "CVaR_Ann":  clean(vals["CVaR"] * sqrt252) if vals["CVaR"] is not None else None,
            }
        return out

    return {
        "capm":    clean(capm),
        "scatter": scatter,
        "var":     clean(annualize_var(var_95)),
        "var_99":  clean(annualize_var(var_99)),
    }


def compute_signals(tech_records: list) -> dict:
    if len(tech_records) < 2:
        return {"signals": []}
    last, prev = tech_records[-1], tech_records[-2]

    data_dict = {
        "RSI":            last.get("RSI")       or 50,
        "MACD_Hist":      last.get("MACD_Hist") or 0,
        "MACD_Hist_Prev": prev.get("MACD_Hist") or 0,
        "Close":          last.get("Close")     or 0,
        "BB_Upper":       last.get("BB_Upper")  or 0,
        "BB_Lower":       last.get("BB_Lower")  or 0,
    }
    signals = generate_signals(data_dict)

    # Golden Cross / Death Cross (SMA_20 vs SMA_50)
    sma20_last = last.get("SMA_20")
    sma50_last = last.get("SMA_50")
    sma20_prev = prev.get("SMA_20")
    sma50_prev = prev.get("SMA_50")

    if all(v is not None for v in [sma20_last, sma50_last, sma20_prev, sma50_prev]):
        if sma20_last > sma50_last and sma20_prev <= sma50_prev:
            signals.append({"id": "GC", "msg": "GOLDEN CROSS (SMA20 cruza SMA50 al alza)", "type": "Buy"})
        elif sma20_last < sma50_last and sma20_prev >= sma50_prev:
            signals.append({"id": "DC", "msg": "DEATH CROSS (SMA20 cruza SMA50 a la baja)", "type": "Sell"})

    # Stochastic crossover en zonas extremas
    k_last = last.get("Stoch_K")
    d_last = last.get("Stoch_D")
    k_prev = prev.get("Stoch_K")
    d_prev = prev.get("Stoch_D")

    if all(v is not None for v in [k_last, d_last, k_prev, d_prev]):
        if k_last > d_last and k_prev <= d_prev and k_last < 30:
            signals.append({"id": "STOCH", "msg": "CRUCE ESTOCÁSTICO ALCISTA (zona sobreventa)", "type": "Buy"})
        elif k_last < d_last and k_prev >= d_prev and k_last > 70:
            signals.append({"id": "STOCH", "msg": "CRUCE ESTOCÁSTICO BAJISTA (zona sobrecompra)", "type": "Sell"})

    return {"signals": signals}


def compute_portfolio(all_returns: dict, rf_annual: float = 0.045) -> dict:
    if len(all_returns) < 2:
        return {}
    df = pd.DataFrame(all_returns).dropna()
    if df.shape[1] < 2:
        return {}

    n_sim = 10_000
    result = clean(optimize_portfolio(df, n_simulations=n_sim, rf_rate=rf_annual))

    # Recalcular Sharpe para los portafolios óptimos con RF real
    mean_ret = df.mean()
    cov_mat  = df.cov()

    def portfolio_metrics(weights_dict):
        w = np.array(list(weights_dict.values()))
        p_ret  = float(np.sum(mean_ret * w) * 252)
        p_vol  = float(np.sqrt(np.dot(w.T, np.dot(cov_mat.values * 252, w))))
        sharpe = (p_ret - rf_annual) / p_vol if p_vol > 0 else 0.0
        return p_ret, p_vol, sharpe

    for key in ["Max_Sharpe", "Min_Volatility"]:
        if key in result:
            w_dict = result[key].get("Weights", {})
            p_ret, p_vol, sharpe = portfolio_metrics(w_dict)
            result[key]["Return"]     = clean(p_ret)
            result[key]["Volatility"] = clean(p_vol)
            result[key]["Sharpe"]     = clean(sharpe)

    # Frontera eficiente: muestra de 2000 portafolios simulados
    n_assets = df.shape[1]
    frontier_ret, frontier_vol, frontier_sharpe = [], [], []
    np.random.seed(42)
    for _ in range(2000):
        w = np.random.random(n_assets)
        w /= w.sum()
        p_r = float(np.sum(mean_ret * w) * 252)
        p_v = float(np.sqrt(np.dot(w.T, np.dot(cov_mat.values * 252, w))))
        p_s = (p_r - rf_annual) / p_v if p_v > 0 else 0.0
        frontier_ret.append(clean(p_r))
        frontier_vol.append(clean(p_v))
        frontier_sharpe.append(clean(p_s))

    result["Frontier"] = {
        "Returns":    frontier_ret,
        "Volatility": frontier_vol,
        "Sharpe":     frontier_sharpe,
    }
    return result


def compute_benchmark(all_returns: dict, bench_data: pd.DataFrame, rf_annual: float = 0.045) -> dict:
    """
    M8: Comparación portafolio óptimo vs benchmark.
    Calcula curvas de retorno acumulado (base 100), drawdown,
    Alpha de Jensen, Beta, Tracking Error, Information Ratio, Max Drawdown.
    """
    if len(all_returns) < 2 or bench_data is None:
        return {}

    try:
        df = pd.DataFrame(all_returns).dropna()
        if df.shape[1] < 2:
            return {}

        # Pesos del portafolio Max Sharpe (10000 sims para cumplir requisito Markowitz)
        port_result = optimize_portfolio(df, n_simulations=10_000, rf_rate=rf_annual)
        weights = np.array(list(port_result["Max_Sharpe"]["Weights"].values()))
        port_ret_series = df.dot(weights)  # retornos diarios del portafolio

        # Alinear con benchmark
        bench_simple, _ = calculate_returns(bench_data)
        common = port_ret_series.index.intersection(bench_simple.index)
        if len(common) < 20:
            return {}

        port_aligned  = port_ret_series.loc[common]
        bench_aligned = bench_simple.loc[common]
        dates = [str(d.date()) if hasattr(d, 'date') else str(d) for d in common]

        # Curvas acumuladas base 100
        port_cum  = (1 + port_aligned).cumprod() * 100
        bench_cum = (1 + bench_aligned).cumprod() * 100

        # Drawdown
        def drawdown_series(ret_series):
            cum = (1 + ret_series).cumprod()
            rolling_max = cum.cummax()
            dd = (cum - rolling_max) / rolling_max
            return dd

        port_dd  = drawdown_series(port_aligned)
        bench_dd = drawdown_series(bench_aligned)

        # Métricas anualizadas
        sqrt252 = math.sqrt(252)

        p_ret_ann  = float((1 + port_aligned.mean()) ** 252 - 1)
        p_vol_ann  = float(port_aligned.std() * sqrt252)
        p_sharpe   = (p_ret_ann - rf_annual) / p_vol_ann if p_vol_ann > 0 else 0.0
        p_max_dd   = float(port_dd.min())

        b_ret_ann  = float((1 + bench_aligned.mean()) ** 252 - 1)
        b_vol_ann  = float(bench_aligned.std() * sqrt252)
        b_sharpe   = (b_ret_ann - rf_annual) / b_vol_ann if b_vol_ann > 0 else 0.0
        b_max_dd   = float(bench_dd.min())

        # Beta y Alpha de Jensen
        slope, intercept, r_val, _, _ = scipy_stats.linregress(bench_aligned, port_aligned)
        beta_port   = float(slope)
        alpha_daily = float(intercept)
        alpha_ann   = float((1 + alpha_daily) ** 252 - 1)

        # Tracking Error y Information Ratio
        active_ret  = port_aligned - bench_aligned
        te_ann      = float(active_ret.std() * sqrt252)
        ir          = float(active_ret.mean() * 252 / te_ann) if te_ann > 0 else 0.0

        return {
            "Dates":          dates,
            "Port_Cum":       clean(port_cum.tolist()),
            "Bench_Cum":      clean(bench_cum.tolist()),
            "Port_DD":        clean(port_dd.tolist()),
            "Bench_DD":       clean(bench_dd.tolist()),
            "Port": {
                "Ann_Return":    clean(p_ret_ann),
                "Ann_Volatility":clean(p_vol_ann),
                "Sharpe":        clean(p_sharpe),
                "Max_Drawdown":  clean(p_max_dd),
            },
            "Bench": {
                "Ann_Return":    clean(b_ret_ann),
                "Ann_Volatility":clean(b_vol_ann),
                "Sharpe":        clean(b_sharpe),
                "Max_Drawdown":  clean(b_max_dd),
            },
            "Jensen_Alpha":     clean(alpha_ann),
            "Beta":             clean(beta_port),
            "Tracking_Error":   clean(te_ann),
            "Information_Ratio":clean(ir),
            "R_Squared":        clean(r_val ** 2),
        }
    except Exception as e:
        print(f"      ! compute_benchmark: {e}", flush=True)
        return {}


# ── Ejecución principal ───────────────────────────────────────────────────────
def main():
    # ── CLI: argumentos validados por Pydantic ────────────────────────────────
    parser = argparse.ArgumentParser(
        description="RiskLab USTA — Generador de datos para dashboard.html",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python generate_data.py\n"
            "  python generate_data.py --start-date 2023-01-01 --end-date 2024-12-31\n"
            "  python generate_data.py --confidence 0.99 --n-simulations 50000\n"
            "  python generate_data.py --start-date 2022-01-01 --end-date 2024-06-30 "
            "--confidence 0.97 --n-simulations 20000\n"
        ),
    )
    parser.add_argument("--start-date",     metavar="YYYY-MM-DD", default=None,
                        help="Fecha inicio del análisis (>=2000-01-01).")
    parser.add_argument("--end-date",       metavar="YYYY-MM-DD", default=None,
                        help="Fecha fin del análisis (no puede ser futura).")
    parser.add_argument("--confidence",     type=float, default=0.95, metavar="N",
                        help="Nivel de confianza VaR [0.80–0.99] (default: 0.95).")
    parser.add_argument("--n-simulations",  type=int,   default=10_000, metavar="N",
                        help="Iteraciones Monte Carlo [1 000–100 000] (default: 10 000).")
    parser.add_argument("--output",         default=str(OUT_FILE), metavar="FILE",
                        help=f"Archivo JS de salida (default: {OUT_FILE.name}).")
    args = parser.parse_args()

    # Validar con Pydantic — mismo contrato que los endpoints FastAPI
    try:
        cfg = GeneratorConfig(
            start_date    = args.start_date,
            end_date      = args.end_date,
            confidence    = args.confidence,
            n_simulations = args.n_simulations,
            output        = args.output,
        )
    except Exception as exc:
        print(f"\n  ERROR de validación de parámetros:\n  {exc}\n", file=sys.stderr)
        parser.print_usage(sys.stderr)
        sys.exit(1)

    out_path = Path(cfg.output)

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  RiskLab USTA - Generador de datos")
    print(f"  Tickers       : {', '.join(TICKERS)}")
    print(f"  Benchmark     : {BENCHMARK}")
    print(f"  Período       : {cfg.start_date or 'últimos 2 años'} → {cfg.end_date or 'hoy'}")
    print(f"  Confianza VaR : {cfg.confidence*100:.0f}%")
    print(f"  Iteraciones MC: {cfg.n_simulations:,}")
    print(f"  Salida        : {out_path.name}")
    print(f"{sep}\n")

    date_kwargs = {}
    if cfg.start_date and cfg.end_date:
        date_kwargs = {"start_date": cfg.start_date, "end_date": cfg.end_date}

    # Tasa libre de riesgo
    print("[0/4] Obteniendo tasa libre de riesgo (^IRX)...")
    rf_annual, rf_source = get_rf_rate()
    print(f"      RF = {rf_annual*100:.2f}% | Fuente: {rf_source}")

    output = {
        "generated_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickers":       TICKERS,
        "rf_rate":       rf_annual,
        "rf_source":     rf_source,
        "var_confidence":cfg.confidence,
        "var_n_sims":    cfg.n_simulations,
        "date_range": {
            "start": cfg.start_date,
            "end":   cfg.end_date,
        },
        "technical":  {},
        "returns":    {},
        "volatility": {},
        "risk":       {},
        "signals":    {},
        "portfolio":  {},
        "benchmark":  {},
    }

    # 1. Benchmark
    print(f"\n[1/4] Descargando benchmark ({BENCHMARK})...")
    bench_data = get_historical_data(BENCHMARK, **date_kwargs)
    if bench_data is None:
        print("      ! No se pudo descargar el benchmark.")
    else:
        print("      OK")

    # 2. Cálculos por ticker
    print(f"\n[2/4] Calculando datos por ticker...\n")
    all_ret = {}

    for ticker in TICKERS:
        print(f"  > {ticker}", flush=True)

        data = get_historical_data(ticker, **date_kwargs)
        if data is None:
            print(f"      ! Sin datos - omitido.\n")
            continue

        # Técnico
        print(f"      Tecnico ...", end=" ", flush=True)
        tech = compute_technical(data)
        output["technical"][ticker] = tech
        print("OK", end="  ", flush=True)

        # Retornos
        print("Retornos ...", end=" ", flush=True)
        output["returns"][ticker] = compute_returns(data)
        print("OK", end="  ", flush=True)

        # GARCH
        print("GARCH ...", end=" ", flush=True)
        try:
            output["volatility"][ticker] = compute_volatility(data)
            print("OK", end="  ", flush=True)
        except Exception as e:
            print(f"! ({e})", end="  ", flush=True)
            output["volatility"][ticker] = {}

        # CAPM + VaR (usa confidence y n_simulations validados)
        print("CAPM+VaR ...", end=" ", flush=True)
        if bench_data is not None:
            output["risk"][ticker] = compute_risk(
                data, bench_data, rf_annual,
                confidence=cfg.confidence,
                n_simulations=cfg.n_simulations,
            )
        else:
            output["risk"][ticker] = {}
        print("OK", end="  ", flush=True)

        # Señales
        output["signals"][ticker] = compute_signals(tech)
        print("Senales OK")

        # Guardar retornos para portafolio
        simple, _ = calculate_returns(data)
        all_ret[ticker] = simple

        print()

    # 3. Portafolio
    print(f"[3/4] Optimizando portafolio (10 000 simulaciones)...")
    try:
        output["portfolio"] = compute_portfolio(all_ret, rf_annual)
        print("      OK\n")
    except Exception as e:
        print(f"      ! Error: {e}\n")
        output["portfolio"] = {}

    # 4. Benchmark M8
    print(f"[4/4] Calculando metricas M8 (portfolio vs benchmark)...")
    try:
        output["benchmark"] = compute_benchmark(all_ret, bench_data, rf_annual)
        print("      OK\n")
    except Exception as e:
        print(f"      ! Error: {e}\n")
        output["benchmark"] = {}

    # 5. Guardar JS
    js_content = (
        "// Generado automaticamente por generate_data.py — NO editar manualmente\n"
        "window.RISKLAB_DATA = "
        + json.dumps(output, ensure_ascii=False, default=str)
        + ";"
    )
    out_path.write_text(js_content, encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"{'=' * 55}")
    print(f"  data.js actualizado")
    print(f"  Tamano    : {size_kb:.1f} KB")
    print(f"  Generado  : {output['generated_at']}")
    print(f"  Tickers OK: {list(output['technical'].keys())}")
    print(f"{'=' * 55}\n")
    print("  Abre dashboard.html en tu navegador para ver los datos.\n")


if __name__ == "__main__":
    main()
