"""
RiskLab USTA — FastAPI Backend
Pydantic validation + dependency injection + date-range & VaR parameters.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy import stats as scipy_stats
from starlette.responses import Response

from api.data import get_historical_data
from api.logic import (
    calculate_bollinger_bands,
    calculate_capm,
    calculate_ema,
    calculate_macd,
    calculate_returns,
    calculate_rsi,
    calculate_sma,
    calculate_stochastic,
    calculate_var_cvar,
    fit_garch_models,
    generate_signals,
    get_descriptive_stats,
    optimize_portfolio,
    perform_normality_tests,
)

load_dotenv()

# ── Importar funciones de cómputo completo desde generate_data.py ─────────────
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from generate_data import (  # type: ignore
        compute_technical,
        compute_returns,
        compute_volatility,
        compute_risk as _compute_risk_full,
        compute_portfolio,
        compute_benchmark,
        get_rf_rate,
    )
    _FULL_COMPUTE_OK = True
except Exception:
    _FULL_COMPUTE_OK = False

app = FastAPI(
    title="RiskLab USTA API",
    version="2.0.0",
    description="Plataforma de análisis cuantitativo de riesgo financiero — USTA",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Configuración centralizada ───────────────────────────────────────────────

class AppConfig(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    benchmark: str = "^GSPC"
    default_rf: float = 0.04
    model_config = {"frozen": True}


@lru_cache(maxsize=1)
def _build_app_config() -> AppConfig:
    raw = os.getenv("PORTFOLIO_TICKERS", "NU,AMZN,SONY,XOM,WPM")
    return AppConfig(
        tickers=[t.strip() for t in raw.split(",") if t.strip()],
        benchmark=os.getenv("BENCHMARK_TICKER", "^GSPC"),
        default_rf=float(os.getenv("RISK_FREE_RATE", "0.04")),
    )


def get_app_config() -> AppConfig:
    return _build_app_config()


ConfigDep = Annotated[AppConfig, Depends(get_app_config)]


# ─── Rango de fechas disponible ───────────────────────────────────────────────
# Coincide con el período de datos generado en data.js
DATA_MIN_DATE = date(2024, 4, 17)   # inicio de datos disponibles
DATA_MAX_DATE = date.today()         # no se pueden solicitar datos futuros


# ─── Dependencia: rango de fechas validado ────────────────────────────────────

def validate_date_range(
    start_date: date | None = Query(
        None,
        description="Fecha de inicio (YYYY-MM-DD). Mínimo: 2024-04-17.",
    ),
    end_date: date | None = Query(
        None,
        description="Fecha de cierre (YYYY-MM-DD). No puede ser futura.",
    ),
) -> dict:
    today = date.today()

    if end_date and end_date > today:
        raise HTTPException(
            status_code=422,
            detail=f"end_date no puede ser una fecha futura (hoy: {today}).",
        )
    if start_date and start_date < DATA_MIN_DATE:
        raise HTTPException(
            status_code=422,
            detail=f"start_date no puede ser anterior a {DATA_MIN_DATE} (inicio de datos disponibles).",
        )
    if start_date and end_date:
        if end_date <= start_date:
            raise HTTPException(
                status_code=422,
                detail="end_date debe ser posterior a start_date.",
            )
        if (end_date - start_date).days < 30:
            raise HTTPException(
                status_code=422,
                detail="El período mínimo de análisis es de 30 días.",
            )
    if bool(start_date) != bool(end_date):
        raise HTTPException(
            status_code=422,
            detail="Debes especificar ambas fechas o ninguna.",
        )

    return {
        "start_date": str(start_date) if start_date else None,
        "end_date":   str(end_date)   if end_date   else None,
    }


DateRangeDep = Annotated[dict, Depends(validate_date_range)]


# ─── Dependencia: parámetros de VaR validados ────────────────────────────────

def validate_var_params(
    confidence: float = Query(
        0.95, ge=0.80, le=0.99,
        description="Nivel de confianza para VaR (0.80–0.99).",
    ),
    n_simulations: int = Query(
        10_000, ge=1_000, le=100_000,
        description="Iteraciones Monte Carlo (1,000–100,000).",
    ),
) -> dict:
    return {"confidence": confidence, "n_simulations": n_simulations}


VaRParamsDep = Annotated[dict, Depends(validate_var_params)]


# ─── Serialización JSON segura ────────────────────────────────────────────────

def _safe_json(obj):
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _safe_json(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "item"):
        return _safe_json(obj.item())
    return obj


def _sv(v):
    """Safe float — convierte NaN/Inf a None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None


def json_response(data) -> Response:
    if isinstance(data, pd.DataFrame):
        body = data.replace([np.inf, -np.inf], np.nan).to_json(orient="records")
    else:
        body = json.dumps(_safe_json(data))
    return Response(content=body, media_type="application/json")


# ─── Helper: datos técnicos completos (M1 / M7) ───────────────────────────────

def _build_technical_records(ticker: str, start_date=None, end_date=None):
    data = get_historical_data(ticker, start_date=start_date, end_date=end_date)
    if data is None:
        return None

    df = data.copy()
    df["Date"]        = df["Date"].dt.strftime("%Y-%m-%d")
    df["SMA_20"]      = calculate_sma(data, 20)
    df["SMA_50"]      = calculate_sma(data, 50)   # necesario para Golden/Death Cross
    df["EMA_20"]      = calculate_ema(data)
    df["RSI"]         = calculate_rsi(data)

    ml, sl, hist      = calculate_macd(data)
    df["MACD_Line"]   = ml
    df["MACD_Signal"] = sl
    df["MACD_Hist"]   = hist

    bb_up, bb_low     = calculate_bollinger_bands(data)
    df["BB_Upper"]    = bb_up
    df["BB_Lower"]    = bb_low

    sk, sd            = calculate_stochastic(data)
    df["Stoch_K"]     = sk
    df["Stoch_D"]     = sd

    df = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(df.to_json(orient="records"))


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", summary="Estado de la API")
def read_root():
    return {
        "message": "Bienvenido a RiskLab USTA API v2.0",
        "status": "ok",
        "data_range": {
            "min": str(DATA_MIN_DATE),
            "max": str(DATA_MAX_DATE),
        },
    }


@app.get("/api/v1/technical/{ticker}", summary="Análisis técnico — M1 / M7")
def get_technical_analysis(
    ticker: str,
    dates: DateRangeDep,
    _cfg: ConfigDep,
):
    try:
        records = _build_technical_records(ticker, **dates)
        if records is None:
            raise HTTPException(404, f"No se encontraron datos para '{ticker}'.")
        return json_response(records)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/returns/{ticker}", summary="Estadísticas de retornos — M2")
def get_returns_analysis(
    ticker: str,
    dates: DateRangeDep,
    _cfg: ConfigDep,
):
    try:
        data = get_historical_data(ticker, **dates)
        if data is None:
            raise HTTPException(404, f"No se encontraron datos para '{ticker}'.")

        simple_ret, log_ret = calculate_returns(data)
        clean_ret = simple_ret.replace([np.inf, -np.inf], np.nan).dropna()

        # ── Q-Q data ──────────────────────────────────────────────────────────
        sorted_ret = np.sort(clean_ret.values)
        n = len(sorted_ret)
        probs = np.linspace(0.01, 0.99, n)
        theoretical_q = scipy_stats.norm.ppf(probs)
        empirical_q   = np.interp(probs, np.linspace(0, 1, n), sorted_ret)

        # ── Stylized facts ────────────────────────────────────────────────────
        skew_val = _sv(float(clean_ret.skew()))
        kurt_val = _sv(float(clean_ret.kurtosis()))   # excess kurtosis

        lb_stat = lb_pval = vol_clustering = None
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_res       = acorr_ljungbox(clean_ret ** 2, lags=[10], return_df=True)
            lb_stat      = _sv(float(lb_res["lb_stat"].iloc[0]))
            lb_pval      = _sv(float(lb_res["lb_pvalue"].iloc[0]))
            vol_clustering = bool(lb_pval is not None and lb_pval < 0.05)
        except Exception:
            pass

        payload = {
            "ticker":    ticker,
            "stats":     get_descriptive_stats(simple_ret),
            "normality": perform_normality_tests(simple_ret),
            "plot_data": {
                "Simple_Returns": _safe_json(clean_ret.fillna(0).tolist()),
                "Log_Returns":    _safe_json(log_ret.replace([np.inf, -np.inf], np.nan).fillna(0).tolist()),
                "Dates":          data["Date"].iloc[1:].dt.strftime("%Y-%m-%d").tolist(),
            },
            "qq_data": {
                "Theoretical": _safe_json(theoretical_q.tolist()),
                "Empirical":   _safe_json(empirical_q.tolist()),
            },
            "stylized_facts": {
                "Skewness":        skew_val,
                "Excess_Kurtosis": kurt_val,
                "LB_Stat":         lb_stat,
                "LB_Pvalue":       lb_pval,
                "Vol_Clustering":  vol_clustering,
                "Neg_Skew":        bool(skew_val is not None and skew_val < 0),
                "Fat_Tails":       bool(kurt_val is not None and kurt_val > 1.0),
            },
        }
        return json_response(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/volatility/{ticker}", summary="Modelos ARCH/GARCH — M3")
def get_volatility_analysis(
    ticker: str,
    dates: DateRangeDep,
    _cfg: ConfigDep,
):
    try:
        data = get_historical_data(ticker, **dates)
        if data is None:
            raise HTTPException(404, f"No se encontraron datos para '{ticker}'.")
        _, log_ret = calculate_returns(data)

        result = fit_garch_models(log_ret)

        # Residuos estandarizados y pronóstico a 10 días
        try:
            from arch import arch_model  # type: ignore
            scaled = log_ret * 100
            m   = arch_model(scaled, vol="GARCH", p=1, q=1)
            res = m.fit(disp="off")
            std_resid    = res.std_resid.dropna().tolist()
            forecast     = res.forecast(horizon=10)
            vol_forecast = (np.sqrt(forecast.variance.iloc[-1].values) / 100).tolist()

            result["Residuals"] = {
                "Std_Residuals": _safe_json(std_resid[-500:]),
                "JB_Stat":       _sv(float(scipy_stats.jarque_bera(std_resid)[0])),
                "JB_Pvalue":     _sv(float(scipy_stats.jarque_bera(std_resid)[1])),
                "Normal":        bool(scipy_stats.jarque_bera(std_resid)[1] > 0.05),
            }
            result["Forecast_10d"] = _safe_json(vol_forecast)
        except Exception:
            pass

        return json_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/risk/{ticker}", summary="CAPM, VaR, CVaR y dispersión — M4 / M5")
def get_risk_analysis(
    ticker: str,
    dates: DateRangeDep,
    var_p: VaRParamsDep,
    config: ConfigDep,
):
    try:
        data       = get_historical_data(ticker, **dates)
        bench_data = get_historical_data(config.benchmark, **dates)

        if data is None or bench_data is None:
            raise HTTPException(
                404,
                "No se encontraron datos para el activo o el benchmark. "
                f"Verifica que el rango de fechas ({dates.get('start_date', 'N/A')} – "
                f"{dates.get('end_date', 'N/A')}) tenga datos disponibles.",
            )

        ret,       _ = calculate_returns(data)
        bench_ret, _ = calculate_returns(bench_data)
        common = ret.index.intersection(bench_ret.index)

        if len(common) >= 10:
            capm_stats = calculate_capm(ret.loc[common], bench_ret.loc[common], config.default_rf)
            # Scatter: puntos alineados + línea de regresión
            asset_vals = _safe_json(ret.loc[common].tolist())
            bench_vals = _safe_json(bench_ret.loc[common].tolist())
            x_min = float(min(b for b in bench_vals if b is not None))
            x_max = float(max(b for b in bench_vals if b is not None))
            reg_x = [x_min, x_max]
            reg_y = [_sv(capm_stats["Alpha"] + capm_stats["Beta"] * x) for x in reg_x]
            scatter = {
                "Asset":     asset_vals,
                "Benchmark": bench_vals,
                "Reg_X":     reg_x,
                "Reg_Y":     reg_y,
            }
        else:
            capm_stats = {
                "Beta": 1.0, "Alpha": 0.0, "R_Squared": 0.0,
                "Expected_Return_Daily": 0.0, "Expected_Return_Annual": 0.0,
                "Classification": "Sin datos suficientes",
            }
            scatter = {"Asset": [], "Benchmark": [], "Reg_X": [], "Reg_Y": []}

        # VaR al nivel de confianza solicitado + VaR 99% de referencia
        sqrt252 = math.sqrt(252)

        def _annualize(v_dict: dict) -> dict:
            out = {}
            for method, vals in v_dict.items():
                if not isinstance(vals, dict):
                    continue
                out[method] = {
                    "VaR":      vals["VaR"],
                    "CVaR":     vals["CVaR"],
                    "VaR_Ann":  _sv(vals["VaR"]  * sqrt252) if vals["VaR"]  is not None else None,
                    "CVaR_Ann": _sv(vals["CVaR"] * sqrt252) if vals["CVaR"] is not None else None,
                }
            return out

        var_95  = _annualize(calculate_var_cvar(ret, confidence=var_p["confidence"],        n_simulations=var_p["n_simulations"]))
        var_99  = _annualize(calculate_var_cvar(ret, confidence=min(var_p["confidence"] + 0.04, 0.99), n_simulations=var_p["n_simulations"]))

        return json_response({
            "capm":    _safe_json(capm_stats),
            "scatter": scatter,
            "var":     var_95,
            "var_99":  var_99,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/portfolio/optimize", summary="Optimización de portafolio Markowitz — M6")
def get_portfolio_optimization(
    dates: DateRangeDep,
    config: ConfigDep,
):
    try:
        all_rets: dict[str, pd.Series] = {}
        for t in config.tickers:
            data = get_historical_data(t, **dates)
            if data is not None:
                ret, _ = calculate_returns(data)
                all_rets[t] = ret

        if len(all_rets) < 2:
            raise HTTPException(500, "No hay suficientes activos con datos disponibles.")

        df_rets = pd.DataFrame(all_rets).dropna()
        return json_response(optimize_portfolio(df_rets))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/signals/{ticker}", summary="Señales técnicas automáticas — M7")
def get_asset_signals(
    ticker: str,
    dates: DateRangeDep,
    _cfg: ConfigDep,
):
    try:
        records = _build_technical_records(ticker, **dates)
        if not records or len(records) < 2:
            raise HTTPException(404, f"Datos insuficientes para '{ticker}'.")

        last, prev = records[-1], records[-2]
        signals = generate_signals({
            "RSI":            last.get("RSI") or 50,
            "MACD_Hist":      last.get("MACD_Hist") or 0,
            "MACD_Hist_Prev": prev.get("MACD_Hist") or 0,
            "Close":          last.get("Close") or 0,
            "BB_Upper":       last.get("BB_Upper") or 0,
            "BB_Lower":       last.get("BB_Lower") or 0,
        })
        return {"ticker": ticker, "signals": signals}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/all", summary="Todos los módulos del dashboard en una sola llamada")
async def get_all_dashboard_data(
    dates: DateRangeDep,
    var_p: VaRParamsDep,
    config: ConfigDep,
):
    """
    Descarga datos de Yahoo Finance y calcula todos los módulos.
    Retorna la misma estructura que window.RISKLAB_DATA en data.js.
    Puede tardar 30-90 segundos.
    """
    if not _FULL_COMPUTE_OK:
        raise HTTPException(500, "Módulos de cómputo no disponibles (generate_data.py).")

    date_kwargs = {k: v for k, v in dates.items() if v is not None}

    rf_annual, rf_source = get_rf_rate()

    output: dict = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickers":       config.tickers,
        "rf_rate":       rf_annual,
        "rf_source":     rf_source,
        "var_confidence": var_p["confidence"],
        "var_n_sims":     var_p["n_simulations"],
        "date_range": {
            "start": dates.get("start_date"),
            "end":   dates.get("end_date"),
        },
        "technical":  {},
        "returns":    {},
        "volatility": {},
        "risk":       {},
        "signals":    {},
        "portfolio":  {},
        "benchmark":  {},
    }

    bench_data = get_historical_data(config.benchmark, **date_kwargs)
    all_ret: dict = {}

    for ticker in config.tickers:
        data = get_historical_data(ticker, **date_kwargs)
        if data is None:
            continue
        try:
            output["technical"][ticker] = compute_technical(data)
        except Exception:
            output["technical"][ticker] = []
        try:
            output["returns"][ticker] = compute_returns(data)
        except Exception:
            output["returns"][ticker] = {}
        try:
            output["volatility"][ticker] = compute_volatility(data)
        except Exception:
            output["volatility"][ticker] = {}
        try:
            if bench_data is not None:
                output["risk"][ticker] = _compute_risk_full(
                    data, bench_data, rf_annual,
                    confidence=var_p["confidence"],
                    n_simulations=var_p["n_simulations"],
                )
        except Exception:
            output["risk"][ticker] = {}
        try:
            simple, _ = calculate_returns(data)
            all_ret[ticker] = simple
        except Exception:
            pass

    try:
        output["portfolio"] = compute_portfolio(all_ret, rf_annual)
    except Exception:
        output["portfolio"] = {}
    try:
        output["benchmark"] = compute_benchmark(all_ret, bench_data, rf_annual)
    except Exception:
        output["benchmark"] = {}

    return json_response(output)
