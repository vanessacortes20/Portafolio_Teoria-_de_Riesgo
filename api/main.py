"""
RiskLab USTA — FastAPI Backend
Pydantic validation + dependency injection + date-range & VaR parameters.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date
from functools import lru_cache
from typing import Annotated

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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


# ─── Configuración centralizada (inyectada como dependencia) ──────────────────

class AppConfig(BaseModel):
    """Application-level configuration loaded once from environment variables."""
    tickers: list[str] = Field(default_factory=list)
    benchmark: str = "^GSPC"
    default_rf: float = 0.04

    model_config = {"frozen": True}


@lru_cache(maxsize=1)
def _build_app_config() -> AppConfig:
    raw = os.getenv("PORTFOLIO_TICKERS", "NU,MELI,SONY,XOM,WPM")
    return AppConfig(
        tickers=[t.strip() for t in raw.split(",") if t.strip()],
        benchmark=os.getenv("BENCHMARK_TICKER", "^GSPC"),
        default_rf=float(os.getenv("RISK_FREE_RATE", "0.04")),
    )


def get_app_config() -> AppConfig:
    return _build_app_config()


ConfigDep = Annotated[AppConfig, Depends(get_app_config)]


# ─── Dependencia: rango de fechas validado ────────────────────────────────────

def validate_date_range(
    start_date: date | None = Query(
        None,
        description="Fecha de inicio del análisis (YYYY-MM-DD). "
                    "Si se omite se usan los últimos 2 años.",
    ),
    end_date: date | None = Query(
        None,
        description="Fecha de cierre del análisis (YYYY-MM-DD). "
                    "Si se omite se usa la fecha de hoy.",
    ),
) -> dict:
    today = date.today()

    if end_date and end_date > today:
        raise HTTPException(
            status_code=422,
            detail="end_date no puede ser una fecha futura.",
        )
    if start_date and start_date < date(2000, 1, 1):
        raise HTTPException(
            status_code=422,
            detail="start_date no puede ser anterior al año 2000.",
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

    return {
        "start_date": str(start_date) if start_date else None,
        "end_date":   str(end_date)   if end_date   else None,
    }


DateRangeDep = Annotated[dict, Depends(validate_date_range)]


# ─── Dependencia: parámetros de VaR validados ────────────────────────────────

def validate_var_params(
    confidence: float = Query(
        0.95,
        ge=0.80,
        le=0.99,
        description="Nivel de confianza para VaR (entre 0.80 y 0.99).",
    ),
    n_simulations: int = Query(
        10_000,
        ge=1_000,
        le=100_000,
        description="Número de iteraciones Monte Carlo (entre 1,000 y 100,000).",
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


def json_response(data) -> Response:
    if isinstance(data, pd.DataFrame):
        body = data.replace([np.inf, -np.inf], np.nan).to_json(orient="records")
    else:
        body = json.dumps(_safe_json(data))
    return Response(content=body, media_type="application/json")


# ─── Helper interno: datos técnicos completos ─────────────────────────────────

def _build_technical_records(ticker: str, start_date=None, end_date=None):
    data = get_historical_data(ticker, start_date=start_date, end_date=end_date)
    if data is None:
        return None

    df = data.copy()
    df["Date"]     = df["Date"].dt.strftime("%Y-%m-%d")
    df["SMA_20"]   = calculate_sma(data)
    df["EMA_20"]   = calculate_ema(data)
    df["RSI"]      = calculate_rsi(data)

    ml, sl, hist   = calculate_macd(data)
    df["MACD_Line"]   = ml
    df["MACD_Signal"] = sl
    df["MACD_Hist"]   = hist

    bb_up, bb_low  = calculate_bollinger_bands(data)
    df["BB_Upper"] = bb_up
    df["BB_Lower"] = bb_low

    sk, sd         = calculate_stochastic(data)
    df["Stoch_K"]  = sk
    df["Stoch_D"]  = sd

    df = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(df.to_json(orient="records"))


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", summary="Estado de la API")
def read_root():
    return {"message": "Bienvenido a RiskLab USTA API v2.0", "status": "ok"}


@app.get("/api/v1/technical/{ticker}", summary="Análisis técnico de un activo")
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


@app.get("/api/v1/returns/{ticker}", summary="Estadística de rendimientos")
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
        payload = {
            "ticker":    ticker,
            "stats":     get_descriptive_stats(simple_ret),
            "normality": perform_normality_tests(simple_ret),
            "plot_data": {
                "Simple_Returns": simple_ret.replace([np.inf, -np.inf], np.nan).fillna(0).tolist(),
                "Log_Returns":    log_ret.replace([np.inf, -np.inf], np.nan).fillna(0).tolist(),
                "Dates":          data["Date"].iloc[1:].dt.strftime("%Y-%m-%d").tolist(),
            },
        }
        return json_response(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/volatility/{ticker}", summary="Modelos de volatilidad ARCH/GARCH")
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
        return json_response(fit_garch_models(log_ret))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/risk/{ticker}", summary="CAPM, VaR y CVaR de un activo")
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
            raise HTTPException(404, "No se encontraron datos para el activo o el benchmark.")

        ret,       _ = calculate_returns(data)
        bench_ret, _ = calculate_returns(bench_data)

        common = ret.index.intersection(bench_ret.index)
        capm_stats = (
            calculate_capm(ret.loc[common], bench_ret.loc[common], config.default_rf)
            if len(common) >= 10
            else {
                "Beta": 1.0, "Alpha": 0.0, "R_Squared": 0.0,
                "Expected_Return_Daily": 0.0, "Expected_Return_Annual": 0.0,
                "Classification": "Sin datos suficientes",
            }
        )

        var_stats = calculate_var_cvar(
            ret,
            confidence=var_p["confidence"],
            n_simulations=var_p["n_simulations"],
        )

        return json_response({"capm": capm_stats, "var": var_stats})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/portfolio/optimize", summary="Optimización de portafolio Markowitz")
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


@app.get("/api/v1/signals/{ticker}", summary="Señales técnicas automáticas")
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
