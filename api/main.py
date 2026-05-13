"""
RiskLab USTA — FastAPI Backend
Pydantic validation + dependency injection + date-range & VaR parameters.
"""
from __future__ import annotations

import json
import math
import os
import secrets
import sys
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

import bcrypt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import re as _re
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from scipy import stats as scipy_stats
from starlette.responses import Response

from api.data import get_historical_data
from api.database import (
    create_user,
    get_all_users,
    get_reset_token,
    get_user_by_cedula,
    get_user_by_email,
    get_user_by_id,
    get_user_by_username,
    init_db,
    mark_token_used,
    save_reset_token,
    seed_demo_users,
    update_last_login,
    update_user_password,
)
from api.logic import (
    arch_lm_test,
    calculate_bollinger_bands,
    calculate_capm,
    calculate_ema,
    calculate_ewma_volatility,
    calculate_macd,
    calculate_returns,
    calculate_rsi,
    calculate_sma,
    calculate_stochastic,
    calculate_var_cvar,
    compare_ewma_vs_garch,
    fit_garch_models,
    generate_signals,
    get_descriptive_stats,
    kupiec_test,
    optimize_portfolio,
    optimize_portfolio_target_return,
    perform_normality_tests,
)

load_dotenv()

# ── Auth config ──────────────────────────────────────────────────────────────
_SECRET_KEY = os.getenv("JWT_SECRET", secrets.token_hex(32))
_ALGORITHM  = "HS256"
_TOKEN_TTL  = int(os.getenv("JWT_TTL_MINUTES", "60"))

_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def _hash(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _verify(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def _create_token(data: dict, expires_minutes: int = _TOKEN_TTL) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=expires_minutes)
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def _decode_token(token: str) -> dict:
    return jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])


def get_current_user(token: str = Depends(_oauth2)) -> dict:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="No autenticado.", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = _decode_token(token)
        user_id: int = payload.get("sub")
        if user_id is None:
            raise ValueError
    except (JWTError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token inválido o expirado.", headers={"WWW-Authenticate": "Bearer"})
    user = get_user_by_id(int(user_id))
    if not user or not user["is_active"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario inactivo o no encontrado.")
    return user


def require_admin(current: dict = Depends(get_current_user)) -> dict:
    if current.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Se requiere rol admin.")
    return current


CurrentUser  = Annotated[dict, Depends(get_current_user)]
AdminUser    = Annotated[dict, Depends(require_admin)]

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


@app.on_event("startup")
def _startup():
    # Capa sqlite3 directa (legacy, conservada en Fase 2)
    init_db()
    seed_demo_users(_hash)
    # Capa SQLAlchemy ORM (nueva en Fase 2 — convive con la anterior)
    from api.database_session import init_orm_tables
    init_orm_tables()


# ── Auth schemas ──────────────────────────────────────────────────────────────

_PHONE_RE = _re.compile(r"^\+?[\d\s\-\(\)]{7,20}$")


class RegisterRequest(BaseModel):
    full_name:        str      = Field(..., min_length=2, max_length=100)
    last_name:        str      = Field(..., min_length=2, max_length=100)
    phone:            str      = Field(..., min_length=7, max_length=20)
    cedula:           str      = Field(..., min_length=4, max_length=20)
    email:            EmailStr
    username:         str      = Field(..., min_length=3, max_length=50,
                                       pattern=r"^[a-zA-Z0-9_.\-]+$")
    password:         str      = Field(..., min_length=8)
    confirm_password: str      = Field(..., min_length=8)

    @field_validator("full_name", "last_name", mode="before")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("Este campo no puede estar vacío.")
        return v

    @field_validator("cedula", mode="before")
    @classmethod
    def _strip_cedula(cls, v: str) -> str:
        return str(v).strip()

    @field_validator("phone", mode="before")
    @classmethod
    def _validate_phone(cls, v: str) -> str:
        v = str(v).strip()
        if not _PHONE_RE.match(v):
            raise ValueError("Formato de teléfono inválido (ej: 3001234567).")
        return v

    @model_validator(mode="after")
    def _passwords_match(self) -> "RegisterRequest":
        if self.password != self.confirm_password:
            raise ValueError("Las contraseñas no coinciden.")
        return self


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserProfile(BaseModel):
    id:         int
    username:   str
    email:      str
    full_name:  str
    last_name:  str
    phone:      str
    cedula:     Optional[str]
    role:       str
    is_active:  int
    created_at: str
    last_login: Optional[str]


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token:        str
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password:     str = Field(..., min_length=8)


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=TokenResponse, summary="Registro de nuevo usuario")
def auth_register(body: RegisterRequest):
    if get_user_by_username(body.username):
        raise HTTPException(400, "El nombre de usuario ya está en uso.")
    if get_user_by_email(body.email):
        raise HTTPException(400, "El correo electrónico ya está registrado.")
    if get_user_by_cedula(body.cedula):
        raise HTTPException(400, "La cédula ya está registrada.")
    user = create_user(
        username=body.username,
        email=body.email,
        hashed_password=_hash(body.password),
        full_name=body.full_name,
        last_name=body.last_name,
        phone=body.phone,
        cedula=body.cedula,
    )
    token = _create_token({"sub": str(user["id"]), "role": user["role"]})
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse, summary="Inicio de sesión")
def auth_login(form: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_username(form.username) or get_user_by_email(form.username)
    if not user or not _verify(form.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user["is_active"]:
        raise HTTPException(400, "Cuenta inactiva.")
    update_last_login(user["id"])
    token = _create_token({"sub": str(user["id"]), "role": user["role"]})
    return TokenResponse(access_token=token)


@app.get("/auth/me", response_model=UserProfile, summary="Perfil del usuario autenticado")
def auth_me(current: CurrentUser):
    return {k: v for k, v in current.items() if k != "hashed_password"}


@app.post("/auth/change-password", summary="Cambiar contraseña")
def auth_change_password(body: ChangePasswordRequest, current: CurrentUser):
    if not _verify(body.current_password, current["hashed_password"]):
        raise HTTPException(400, "Contraseña actual incorrecta.")
    update_user_password(current["id"], _hash(body.new_password))
    return {"message": "Contraseña actualizada correctamente."}


@app.post("/auth/reset-password", summary="Solicitar restablecimiento de contraseña")
def auth_reset_request(body: PasswordResetRequest):
    user = get_user_by_email(body.email)
    if not user:
        return {"message": "Si el correo está registrado, recibirás instrucciones."}
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=1)).isoformat(timespec="seconds")
    save_reset_token(user["id"], token, expires)
    return {"message": "Token generado. Úsalo en /auth/reset-password/confirm.", "reset_token": token}


@app.post("/auth/reset-password/confirm", summary="Confirmar restablecimiento con token")
def auth_reset_confirm(body: PasswordResetConfirm):
    record = get_reset_token(body.token)
    if not record:
        raise HTTPException(400, "Token inválido o ya utilizado.")
    if datetime.fromisoformat(record["expires_at"]) < datetime.utcnow():
        raise HTTPException(400, "Token expirado.")
    update_user_password(record["user_id"], _hash(body.new_password))
    mark_token_used(body.token)
    return {"message": "Contraseña restablecida correctamente."}


@app.get("/auth/users", summary="Listar todos los usuarios — solo admin")
def auth_list_users(_admin: AdminUser):
    return get_all_users()


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
DATA_MIN_DATE = date(2020, 1, 1)   # inicio mínimo — Yahoo Finance tiene datos desde 2020
DATA_MAX_DATE = date.today()         # no se pueden solicitar datos futuros


# ─── Dependencia: rango de fechas validado ────────────────────────────────────

def validate_date_range(
    start_date: date | None = Query(
        None,
        description="Fecha de inicio (YYYY-MM-DD). Mínimo: 2020-01-01.",
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


@app.get("/api/v1/volatility/{ticker}", summary="Modelos EWMA + ARCH/GARCH — M3")
def get_volatility_analysis(
    ticker: str,
    dates: DateRangeDep,
    _cfg: ConfigDep,
    lambda_ewma: float = Query(
        0.94, gt=0.0, lt=1.0,
        description="Factor de decaimiento EWMA (0 < λ < 1). Default 0.94 (RiskMetrics).",
    ),
):
    try:
        data = get_historical_data(ticker, **dates)
        if data is None:
            raise HTTPException(404, f"No se encontraron datos para '{ticker}'.")
        _, log_ret = calculate_returns(data)

        result = fit_garch_models(log_ret)

        # Residuos estandarizados, pronóstico a 10 días y diagnóstico ARCH-LM
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
                "ARCH_LM":       _safe_json(arch_lm_test(std_resid, nlags=5)),
            }
            result["Forecast_10d"] = _safe_json(vol_forecast)
        except Exception:
            pass

        # EWMA y comparación contra GARCH(1,1)
        try:
            ewma = calculate_ewma_volatility(log_ret, lambda_=lambda_ewma)
            garch_vol = result.get("GARCH(1,1)", {}).get("Volatility", [])
            garch_last = float(garch_vol[-1]) if garch_vol else None
            comparison = compare_ewma_vs_garch(ewma.get("ewma_last_value"), garch_last)
            result["EWMA"] = _safe_json({
                "lambda":          ewma["lambda"],
                "ewma_volatility": ewma["ewma_volatility"][-500:],
                "ewma_last_value": ewma["ewma_last_value"],
                "ewma_mean":       ewma["ewma_mean"],
                "rolling_30d_avg": ewma["rolling_30d_avg"],
            })
            result["comparison_ewma_garch"] = _safe_json(comparison)
            result["interpretation"] = comparison.get("interpretation")
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

        var_95  = _annualize(calculate_var_cvar(ret, confidence=var_p["confidence"], n_simulations=var_p["n_simulations"]))
        var_99  = _annualize(calculate_var_cvar(ret, confidence=0.99,               n_simulations=var_p["n_simulations"]))

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
        return json_response(optimize_portfolio(df_rets, rf_rate=config.default_rf))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/risk/{ticker}/backtest", summary="Backtesting VaR — Test de Kupiec — M5")
def get_var_backtest(
    ticker: str,
    dates: DateRangeDep,
    var_p: VaRParamsDep,
):
    try:
        data = get_historical_data(ticker, **dates)
        if data is None:
            raise HTTPException(404, f"No se encontraron datos para '{ticker}'.")

        ret, _ = calculate_returns(data)
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        var_result = calculate_var_cvar(ret, confidence=var_p["confidence"], n_simulations=var_p["n_simulations"])

        backtesting = {}
        for method in ("Historico", "Parametrico", "Montecarlo"):
            var_val = var_result[method]["VaR"]
            backtesting[method] = kupiec_test(ret, var_val, var_p["confidence"])

        return json_response({
            "ticker":        ticker,
            "confidence":    var_p["confidence"],
            "backtesting":   backtesting,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/portfolio/target", summary="Portafolio por Rendimiento Objetivo — M6")
def get_portfolio_target_return(
    dates: DateRangeDep,
    config: ConfigDep,
    target_return: float = Query(..., ge=-0.5, le=2.0, description="Rendimiento anual objetivo (ej: 0.15 = 15%)"),
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
        result  = optimize_portfolio_target_return(df_rets, target_return, rf_rate=config.default_rf)
        return json_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/macro", summary="Indicadores macroeconómicos de contexto — M8")
def get_macro_indicators(db = Depends(lambda: None)):
    """Devuelve tasa libre de riesgo, rendimiento del Tesoro a 10 años y retorno YTD del S&P 500.

    Fuente primaria: FRED (DGS3MO, DGS10, CPIAUCSL) con cache transparente en SQLite.
    Fallback: yfinance (^IRX, ^TNX, ^GSPC) si FRED no está disponible o falla.
    Las keys originales (as_of, rf_rate, rf_source, treasury_10y, spx_ytd) se preservan
    para no romper el dashboard. Se agregan campos opcionales con info de cache.
    """
    import yfinance as yf
    from api.database_session import SessionLocal
    from api.services import fred_service as fred

    result: dict = {"as_of": date.today().isoformat()}
    cache_status = {}

    # ---- Tasa libre de riesgo (3 meses) ----
    rf_value = None
    rf_source = None
    if fred.is_available():
        db_local = SessionLocal()
        try:
            rf_data = fred.get_rf_rate_3m(db_local)
            if rf_data and rf_data.get("value_decimal") is not None:
                rf_value = rf_data["value_decimal"]
                rf_source = f"FRED.DGS3MO ({rf_data.get('date','')})"
                cache_status["rf_rate"] = rf_data.get("cache_status")
        finally:
            db_local.close()

    if rf_value is None:
        # Fallback yfinance ^IRX
        try:
            irx_hist = yf.Ticker("^IRX").history(period="5d")
            if not irx_hist.empty:
                rf_value = float(irx_hist["Close"].iloc[-1]) / 100
                rf_source = "^IRX T-Bill 13 sem."
            else:
                rf_value = 0.04
                rf_source = "default"
        except Exception:
            rf_value = 0.04
            rf_source = "default"

    result["rf_rate"] = rf_value
    result["rf_source"] = rf_source

    # ---- Treasury 10Y ----
    t10y_value = None
    t10y_source = None
    if fred.is_available():
        db_local = SessionLocal()
        try:
            t10 = fred.get_treasury_10y(db_local)
            if t10 and t10.get("value_decimal") is not None:
                t10y_value = t10["value_decimal"]
                t10y_source = f"FRED.DGS10 ({t10.get('date','')})"
                cache_status["treasury_10y"] = t10.get("cache_status")
        finally:
            db_local.close()

    if t10y_value is None:
        try:
            tnx_hist = yf.Ticker("^TNX").history(period="5d")
            t10y_value = float(tnx_hist["Close"].iloc[-1]) / 100 if not tnx_hist.empty else None
            t10y_source = "^TNX yfinance"
        except Exception:
            t10y_value = None
            t10y_source = "unavailable"

    result["treasury_10y"] = t10y_value
    result["treasury_10y_source"] = t10y_source

    # ---- S&P 500 YTD (yfinance, no hay equivalente directo en FRED) ----
    try:
        spy_hist = yf.Ticker("^GSPC").history(period="ytd")
        if not spy_hist.empty and len(spy_hist) > 1:
            result["spx_ytd"] = float(spy_hist["Close"].iloc[-1] / spy_hist["Close"].iloc[0] - 1)
        else:
            result["spx_ytd"] = None
    except Exception:
        result["spx_ytd"] = None

    # ---- Inflación CPI (informativo, opcional) ----
    if fred.is_available():
        db_local = SessionLocal()
        try:
            cpi = fred.get_inflation_yoy(db_local)
            if cpi:
                result["inflation_yoy"] = cpi.get("yoy")
                cache_status["inflation_yoy"] = cpi.get("cache_status")
        finally:
            db_local.close()

    # ---- Metadatos de fuente ----
    result["fred_enabled"] = fred.is_available()
    if cache_status:
        result["cache_status"] = cache_status

    return json_response(result)


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
