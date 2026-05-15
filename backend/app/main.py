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

from backend.app.data_yf import get_historical_data
from backend.app.auth_db import (
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
from backend.app.services.logic import (
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
    compare_qp_long_only_vs_short,
    fit_garch_models,
    generate_signals,
    get_descriptive_stats,
    kupiec_test,
    optimize_portfolio,
    optimize_portfolio_qp,
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
# generate_data.py vive en la raíz del proyecto: backend/app → parent.parent.parent
_project_root = Path(__file__).parent.parent.parent
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
    from backend.app.database import init_orm_tables
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

def _build_technical_records(
    ticker: str,
    start_date=None,
    end_date=None,
    sma_short: int = 20,
    sma_long:  int = 50,
    ema_window: int = 20,
    rsi_window: int = 14,
    bb_window:  int = 20,
    bb_std:     float = 2.0,
):
    """Calcula todos los indicadores técnicos del M1.

    Parámetros ajustables (defaults estándar de la industria):
    - sma_short, sma_long: ventanas de las dos medias móviles simples (Cruce Dorado)
    - ema_window: ventana de la media móvil exponencial
    - rsi_window: ventana del RSI (default 14)
    - bb_window, bb_std: parámetros de las Bandas de Bollinger
    """
    data = get_historical_data(ticker, start_date=start_date, end_date=end_date)
    if data is None:
        return None

    df = data.copy()
    df["Date"]        = df["Date"].dt.strftime("%Y-%m-%d")
    df["SMA_20"]      = calculate_sma(data, sma_short)
    df["SMA_50"]      = calculate_sma(data, sma_long)   # necesario para Golden/Death Cross
    df["EMA_20"]      = calculate_ema(data, ema_window)
    df["RSI"]         = calculate_rsi(data, rsi_window)

    ml, sl, hist      = calculate_macd(data)
    df["MACD_Line"]   = ml
    df["MACD_Signal"] = sl
    df["MACD_Hist"]   = hist

    bb_up, bb_low     = calculate_bollinger_bands(data, bb_window, bb_std)
    df["BB_Upper"]    = bb_up
    df["BB_Middle"]   = calculate_sma(data, bb_window)   # banda media = SMA del mismo window
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
    sma_short:  int   = Query(20, ge=2,   le=200, description="Ventana SMA corta (default 20)."),
    sma_long:   int   = Query(50, ge=5,   le=400, description="Ventana SMA larga (default 50)."),
    ema_window: int   = Query(20, ge=2,   le=200, description="Ventana EMA (default 20)."),
    rsi_window: int   = Query(14, ge=2,   le=100, description="Ventana RSI (default 14)."),
    bb_window:  int   = Query(20, ge=5,   le=200, description="Ventana Bollinger (default 20)."),
    bb_std:     float = Query(2.0, gt=0,  le=5,   description="Desviaciones estándar Bollinger (default 2.0)."),
):
    try:
        records = _build_technical_records(
            ticker, **dates,
            sma_short=sma_short, sma_long=sma_long, ema_window=ema_window,
            rsi_window=rsi_window, bb_window=bb_window, bb_std=bb_std,
        )
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
        clean_log = log_ret.replace([np.inf, -np.inf], np.nan).dropna()

        # ── Q-Q data ──────────────────────────────────────────────────────────
        sorted_ret = np.sort(clean_ret.values)
        n = len(sorted_ret)
        probs = np.linspace(0.01, 0.99, n)
        theoretical_q = scipy_stats.norm.ppf(probs)
        empirical_q   = np.interp(probs, np.linspace(0, 1, n), sorted_ret)

        # ── Curva normal superpuesta para el histograma ──────────────────────
        mu_emp    = float(clean_ret.mean())
        sigma_emp = float(clean_ret.std())
        x_range   = np.linspace(clean_ret.min(), clean_ret.max(), 200)
        normal_y  = scipy_stats.norm.pdf(x_range, mu_emp, sigma_emp)

        # ── Boxplot: cuartiles + outliers IQR ────────────────────────────────
        q1, med, q3 = np.percentile(clean_ret, [25, 50, 75])
        iqr         = q3 - q1
        lower_w     = float(max(clean_ret.min(), q1 - 1.5 * iqr))
        upper_w     = float(min(clean_ret.max(), q3 + 1.5 * iqr))
        outliers    = clean_ret[(clean_ret < lower_w) | (clean_ret > upper_w)]

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

        # Efecto apalancamiento (proxy): correlación entre |r_{t-1}| y r_t.
        # Si es negativa, las caídas anteriores se asocian con más volatilidad
        # (clásico leverage de Black 1976).
        leverage_corr: Optional[float] = None
        leverage_present: Optional[bool] = None
        try:
            shifted_abs = clean_ret.abs().shift(1)
            corr_df = pd.concat([shifted_abs, clean_ret], axis=1).dropna()
            leverage_corr = _sv(float(corr_df.iloc[:, 0].corr(corr_df.iloc[:, 1])))
            leverage_present = bool(leverage_corr is not None and leverage_corr < -0.05)
        except Exception:
            pass

        # ── Pruebas de normalidad CON interpretación textual ─────────────────
        normality_raw = perform_normality_tests(simple_ret)
        def _interp_p(p: Optional[float]) -> str:
            if p is None:
                return "no disponible"
            if p < 0.05:
                return f"p={p:.4g} < 0.05 → se rechaza H0 de normalidad"
            return f"p={p:.4g} ≥ 0.05 → no se rechaza la normalidad (no implica que sí lo sea)"

        normality_full = {}
        for test_name, vals in normality_raw.items():
            p_val = vals.get("p_value")
            normality_full[test_name] = {
                "stat":           _sv(vals.get("stat")),
                "p_value":        _sv(p_val),
                "rejects_normal": bool(p_val is not None and p_val < 0.05),
                "interpretation": _interp_p(p_val),
            }

        # Interpretación de hechos estilizados
        notes: list[str] = []
        if kurt_val is not None and kurt_val > 1.0:
            notes.append(f"Curtosis exc. = {kurt_val:.2f} indica colas pesadas (eventos extremos más frecuentes que normal).")
        if skew_val is not None and abs(skew_val) > 0.2:
            sign = "negativa (cola izquierda más pesada — caídas más frecuentes)" if skew_val < 0 else "positiva (cola derecha más pesada)"
            notes.append(f"Asimetría {sign}: {skew_val:.3f}.")
        if vol_clustering:
            notes.append("Ljung-Box sobre r² rechaza independencia → agrupamiento de volatilidad presente (justifica modelar con GARCH en M3).")
        if leverage_present:
            notes.append(f"Efecto apalancamiento detectado: corr(|r_(t-1)|, r_t) = {leverage_corr:.3f} (caídas anteriores asocian con mayor volatilidad).")

        # Estadísticas también para log-rendimientos (la imagen lo prefiere)
        log_stats = get_descriptive_stats(clean_log) if len(clean_log) else {}

        payload = {
            "ticker":    ticker,
            "n_obs":     int(len(clean_ret)),
            "log_returns_justification": (
                "Se usan log-rendimientos como base estadística por su aditividad temporal "
                "(la suma de log-retornos diarios es el log-retorno acumulado), su simetría "
                "frente a subidas y caídas equivalentes, y su buena aproximación a los "
                "retornos simples cuando son pequeños (|r|<5%)."
            ),
            "stats":           get_descriptive_stats(simple_ret),
            "stats_log":       log_stats,
            "normality":       normality_full,
            "normality_legacy": normality_raw,   # compatibilidad hacia atrás
            "plot_data": {
                "Simple_Returns": _safe_json(clean_ret.fillna(0).tolist()),
                "Log_Returns":    _safe_json(log_ret.replace([np.inf, -np.inf], np.nan).fillna(0).tolist()),
                "Dates":          data["Date"].iloc[1:].dt.strftime("%Y-%m-%d").tolist(),
            },
            "normal_curve": {
                "x":     _safe_json(x_range.tolist()),
                "y":     _safe_json(normal_y.tolist()),
                "mu":    _sv(mu_emp),
                "sigma": _sv(sigma_emp),
            },
            "qq_data": {
                "Theoretical": _safe_json(theoretical_q.tolist()),
                "Empirical":   _safe_json(empirical_q.tolist()),
            },
            "boxplot_stats": {
                "min":            _sv(float(clean_ret.min())),
                "q1":             _sv(float(q1)),
                "median":         _sv(float(med)),
                "q3":             _sv(float(q3)),
                "max":            _sv(float(clean_ret.max())),
                "iqr":            _sv(float(iqr)),
                "lower_whisker":  _sv(lower_w),
                "upper_whisker":  _sv(upper_w),
                "n_outliers":     int(len(outliers)),
            },
            "stylized_facts": {
                "Skewness":        skew_val,
                "Excess_Kurtosis": kurt_val,
                "LB_Stat":         lb_stat,
                "LB_Pvalue":       lb_pval,
                "Vol_Clustering":  vol_clustering,
                "Neg_Skew":        bool(skew_val is not None and skew_val < 0),
                "Fat_Tails":       bool(kurt_val is not None and kurt_val > 1.0),
                "Leverage_Corr":   leverage_corr,
                "Leverage_Effect": leverage_present,
                "interpretation":  " ".join(notes) if notes else "Sin desviaciones marcadas frente a la normal.",
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
        clean_log = log_ret.replace([np.inf, -np.inf], np.nan).dropna()

        result = fit_garch_models(log_ret)

        # ── Parámetros GARCH(1,1): ω, α, β + varianza incondicional ──────────
        garch_params: dict = {}
        try:
            from arch import arch_model  # type: ignore
            scaled = log_ret * 100
            m_g    = arch_model(scaled, vol="GARCH", p=1, q=1)
            res_g  = m_g.fit(disp="off")
            params = res_g.params.to_dict()
            omega  = float(params.get("omega", 0))
            alpha  = float(params.get("alpha[1]", 0))
            beta   = float(params.get("beta[1]", 0))
            persistence = alpha + beta
            uncond_var = (omega / (1 - persistence)) if persistence < 1 else None
            garch_params = {
                "omega":               round(omega, 8),
                "alpha":               round(alpha, 6),
                "beta":                round(beta, 6),
                "persistence":         round(persistence, 6),
                "unconditional_var":   round(uncond_var, 8) if uncond_var is not None else None,
                "unconditional_vol":   round(np.sqrt(uncond_var) / 100, 6) if uncond_var is not None else None,
                "mean_reversion":      bool(persistence < 1),
                "interpretation":      (
                    f"Persistencia α+β = {persistence:.4f}. "
                    + ("Hay reversión a la media (α+β<1); la varianza vuelve al nivel incondicional con velocidad 1-α-β."
                       if persistence < 1 else
                       "α+β ≥ 1 → no hay varianza incondicional finita (proceso integrado tipo IGARCH).")
                ),
            }
            result["GARCH(1,1)"]["parameters"] = garch_params
        except Exception:
            pass

        # ── Residuos, pronóstico, diagnósticos JB y ARCH-LM ──────────────────
        try:
            from arch import arch_model  # type: ignore
            scaled = log_ret * 100
            m   = arch_model(scaled, vol="GARCH", p=1, q=1)
            res = m.fit(disp="off")
            std_resid    = res.std_resid.dropna().tolist()
            forecast     = res.forecast(horizon=10)
            vol_forecast = (np.sqrt(forecast.variance.iloc[-1].values) / 100).tolist()

            jb_stat, jb_p = scipy_stats.jarque_bera(std_resid)
            jb_p = float(jb_p)
            jb_interp = (
                f"p={jb_p:.4g} < 0.05 → se rechaza normalidad de residuos (justifica usar t-Student en GARCH si se quiere mayor robustez)."
                if jb_p < 0.05 else
                f"p={jb_p:.4g} ≥ 0.05 → no se rechaza normalidad de residuos estandarizados (modelo capturó bien la heterocedasticidad)."
            )

            result["Residuals"] = {
                "Std_Residuals":  _safe_json(std_resid[-500:]),
                "JB_Stat":        _sv(float(jb_stat)),
                "JB_Pvalue":      _sv(jb_p),
                "Normal":         bool(jb_p > 0.05),
                "JB_Interpretation": jb_interp,
                "ARCH_LM":        _safe_json(arch_lm_test(std_resid, nlags=5)),
            }
            result["Forecast_10d"] = _safe_json(vol_forecast)
        except Exception:
            pass

        # ── EWMA + serie rodante + comparación contra GARCH(1,1) ─────────────
        try:
            ewma = calculate_ewma_volatility(log_ret, lambda_=lambda_ewma)
            garch_vol = result.get("GARCH(1,1)", {}).get("Volatility", [])
            garch_last = float(garch_vol[-1]) if garch_vol else None

            # Volatilidad muestral rodante 30d (serie completa, no solo promedio)
            rolling_30 = clean_log.rolling(window=30).std().dropna()

            comparison = compare_ewma_vs_garch(ewma.get("ewma_last_value"), garch_last)
            result["EWMA"] = _safe_json({
                "lambda":              ewma["lambda"],
                "ewma_volatility":     ewma["ewma_volatility"][-500:],
                "ewma_last_value":     ewma["ewma_last_value"],
                "ewma_mean":           ewma["ewma_mean"],
                "rolling_30d_avg":     ewma["rolling_30d_avg"],
                "rolling_30d_series":  rolling_30.iloc[-500:].tolist(),
            })
            result["comparison_ewma_garch"] = _safe_json(comparison)
            result["interpretation"] = comparison.get("interpretation")

            # Tabla estructurada EWMA vs GARCH(1,1) (formato del profesor)
            result["comparison_table"] = _safe_json({
                "title": "Comparación EWMA vs GARCH(1,1)",
                "rows": [
                    {"aspect": "Parámetros estimados",
                     "EWMA":     "0 (λ fijo o calibrado)",
                     "GARCH":    f"3 (ω={garch_params.get('omega')}, α={garch_params.get('alpha')}, β={garch_params.get('beta')})"
                                 if garch_params else "3 (ω, α, β)"},
                    {"aspect": "Varianza incondicional",
                     "EWMA":  "No definida",
                     "GARCH": f"σ²={garch_params.get('unconditional_var')} → σ={garch_params.get('unconditional_vol')}"
                              if garch_params and garch_params.get('unconditional_var') is not None
                              else "σ² = ω/(1−α−β)"},
                    {"aspect": "Reversión a la media",
                     "EWMA":  "No",
                     "GARCH": "Sí" if (garch_params.get("mean_reversion") if garch_params else True) else "No (α+β≥1)"},
                    {"aspect": "Costo computacional",
                     "EWMA":  "Mínimo (recursión)",
                     "GARCH": "Optimización por máxima verosimilitud"},
                    {"aspect": "Captura asimetría",
                     "EWMA":  "No",
                     "GARCH": "Solo en variantes (EGARCH, GJR)"},
                    {"aspect": "Interpretación",
                     "EWMA":  "Decay exponencial constante",
                     "GARCH": "Estructura paramétrica completa"},
                ],
            })
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
            # Rf desde FRED (con fallback a yfinance/.env) — no hardcodeada
            rf_annual_used, _rf_source_used = _resolve_rf_from_fred_or_fallback(config)
            capm_stats = calculate_capm(ret.loc[common], bench_ret.loc[common], rf_annual_used)
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
    include_qp: bool = Query(
        True,
        description="Incluye optimización QP determinista (long-only y con short permitido).",
    ),
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
        result = optimize_portfolio(df_rets, rf_rate=config.default_rf)

        if include_qp:
            try:
                qp = compare_qp_long_only_vs_short(df_rets, rf_rate=config.default_rf)
                # Campos nuevos sin remover los existentes (Max_Sharpe, Min_Volatility, Correlation)
                result["qp_min_variance"]               = qp["long_only"]["min_variance"]
                result["qp_max_sharpe"]                 = qp["long_only"]["max_sharpe"]
                result["allow_short_false_result"]      = qp["long_only"]
                result["allow_short_true_result"]       = qp["with_short"]
                result["comparison_long_only_vs_short_allowed"] = {
                    "sharpe_gain_with_short":         qp["sharpe_gain_with_short"],
                    "zero_weight_in_long_only":       qp["zero_weight_in_long_only"],
                    "short_positions_when_allowed":   qp["short_positions_when_allowed"],
                    "interpretation":                 qp["interpretation"],
                }
            except Exception as exc:  # pragma: no cover
                result["qp_error"] = str(exc)

        return json_response(result)
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
    from backend.app.database import SessionLocal
    from backend.app.services import fred_service as fred

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


class PredictRequest(BaseModel):
    ticker:   str = Field(..., min_length=1, max_length=15)
    features: list[float] = Field(..., min_length=1, max_length=20,
        description="Vector de features (debe coincidir con el orden documentado del modelo).")
    horizon:  Optional[int] = Field(1, ge=1, le=30,
        description="Horizonte de predicción en días (informativo).")

    @field_validator("features")
    @classmethod
    def _features_finite(cls, v: list[float]) -> list[float]:
        for x in v:
            if not isinstance(x, (int, float)):
                raise ValueError("Todas las features deben ser numéricas")
            if x != x or x in (float("inf"), float("-inf")):  # NaN o inf
                raise ValueError("Las features no pueden ser NaN ni infinito")
        return v


@app.post("/api/v1/predict", summary="Predicción ML direccional buy/hold/sell — M12")
def post_predict(req: PredictRequest):
    """Predice dirección del retorno usando el modelo Singleton.

    El modelo se carga UNA sola vez al primer request (patrón Singleton).
    Cada predicción se persiste en PredictionLog.
    """
    from backend.app.database import SessionLocal
    from backend.app.models.db_models import PredictionLog
    from backend.app.ml.predictor import get_predictor

    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(503, "Modelo ML no disponible. Ejecuta `python -m api.ml.train` para generarlo.")

    expected = predictor.feature_columns()
    if expected and len(req.features) != len(expected):
        raise HTTPException(
            422,
            f"Se esperaban {len(expected)} features ({expected}); recibidas {len(req.features)}.",
        )

    try:
        label_int, probs = predictor.predict(np.asarray(req.features))
    except Exception as exc:
        raise HTTPException(500, f"Error en la predicción: {exc}")

    label_map = predictor.labels()
    label_str = label_map.get(str(label_int), str(label_int))

    # Persistencia en PredictionLog
    db = SessionLocal()
    try:
        log = PredictionLog(
            model_version=predictor.model_version,
            ticker=req.ticker,
            input_features={
                "values": req.features,
                "names":  expected,
                "horizon": req.horizon,
            },
            prediction=float(label_int),
        )
        db.add(log)
        db.commit()
        log_id = log.id
    finally:
        db.close()

    interp = (
        f"Predicción para {req.ticker}: {label_str.upper()}. "
        "Esta señal es probabilística y no constituye recomendación financiera. "
        "No incluye costos de transacción, slippage ni cambios de régimen."
    )

    return json_response({
        "ticker":           req.ticker,
        "prediction":       label_int,
        "prediction_label": label_str,
        "probability":      probs,
        "model_version":    predictor.model_version,
        "features_used":    expected,
        "horizon":          req.horizon,
        "log_id":           log_id,
        "interpretation":   interp,
        "warning":          ("Modelo académico — NO constituye recomendación de inversión "
                             "ni garantiza rentabilidad. Use con criterio profesional."),
    })


@app.get("/api/v1/predict/info", summary="Metadata del modelo ML cargado")
def get_predict_info():
    """Devuelve metadata del modelo, sin requerir input. Útil para docs/dashboard."""
    from backend.app.ml.predictor import get_predictor
    predictor = get_predictor()
    return json_response({
        "is_ready":       predictor.is_ready,
        "model_version":  predictor.model_version,
        "feature_cols":   predictor.feature_columns(),
        "labels":         predictor.labels(),
        "metadata":       predictor.metadata,
    })


class StressScenario(BaseModel):
    name:            str   = Field(..., min_length=1, max_length=80)
    market_drop_pct: Optional[float] = Field(None, ge=-1.0, le=1.0,
        description="Caída del mercado en decimal (ej: -0.20).")
    rate_shock_bp:   Optional[int]   = Field(None, ge=-500, le=500,
        description="Shock de tasa en puntos básicos.")
    vol_multiplier:  Optional[float] = Field(None, gt=0, le=10,
        description="Multiplicador de volatilidad (1.0 = sin cambio).")


class StressRequest(BaseModel):
    weights:    dict[str, float] = Field(..., description="Mapa ticker → peso (deben sumar 1).")
    prices:     dict[str, float] = Field(..., description="Mapa ticker → precio actual.")
    betas:      Optional[dict[str, float]] = Field(None, description="Beta por activo (default 1.0).")
    sigmas:     Optional[dict[str, float]] = Field(None, description="Sigma anual por activo (default 0.20).")
    scenarios:  Optional[list[StressScenario]] = Field(None,
        description="Si no se pasa, se corren los 6 escenarios obligatorios.")

    @model_validator(mode="after")
    def _weights_sum_one(self) -> "StressRequest":
        s = sum(self.weights.values())
        if abs(s - 1.0) > 1e-3:
            raise ValueError(f"weights deben sumar 1; suman {s:.4f}")
        return self


@app.post("/api/v1/stress", summary="Stress testing del portafolio bajo escenarios — M11")
def post_stress(req: StressRequest):
    """Aplica escenarios de mercado, tasa y volatilidad sobre el portafolio."""
    from backend.app.services.stress import StressTester
    try:
        scenarios = [s.model_dump() for s in req.scenarios] if req.scenarios else None
        tester = StressTester(
            weights=req.weights,
            prices=req.prices,
            betas=req.betas,
            sigmas=req.sigmas,
        )
        return json_response(tester.run_all(scenarios))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"Error en stress test: {exc}")


class OptionRequest(BaseModel):
    S:           float = Field(..., gt=0,  description="Precio spot del subyacente.")
    K:           float = Field(..., gt=0,  description="Strike de la opción.")
    T:           float = Field(..., gt=0,  le=30, description="Tiempo a vencimiento en años.")
    r:           float = Field(..., ge=-0.05, le=1, description="Tasa libre de riesgo anual decimal.")
    sigma:       float = Field(..., gt=0,  le=5,   description="Volatilidad anual decimal.")
    option_type: str   = Field("call", description="'call' o 'put'.")

    @field_validator("option_type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("call", "put"):
            raise ValueError("option_type debe ser 'call' o 'put'")
        return v


@app.get("/api/v1/opcion/precio/{ticker}",
         summary="Black-Scholes sobre un activo del portafolio (S y σ obtenidos automáticamente) — M10")
def get_option_price_for_ticker(
    ticker: str,
    K_pct:        float = Query(1.0, gt=0,    description="Strike como fracción del spot (1.0 = ATM)."),
    T:            float = Query(0.25, gt=0, le=10, description="Tiempo a vencimiento en años."),
    option_type:  str   = Query("call", description="'call' o 'put'."),
    sigma_window: int   = Query(60, ge=20, le=252, description="Días para σ histórica anualizada."),
):
    """Valoración Black-Scholes usando el spot real y σ histórica del activo.

    Obtiene precios desde el cache SQLAlchemy (descarga si falta), calcula σ
    como std de log-retornos × √252, toma Rf desde /api/v1/macro y devuelve
    el precio + Greeks + paridad. Cumple el requisito del instructivo:
    *las opciones se valoran sobre los mismos activos del portafolio*.
    """
    from backend.app.database import SessionLocal
    from backend.app.services.options import OptionPricer
    from backend.app.services.price_service import get_prices
    from backend.app.services import fred_service as fred
    import yfinance as yf
    import numpy as np

    if option_type not in ("call", "put"):
        raise HTTPException(422, "option_type debe ser 'call' o 'put'")

    db = SessionLocal()
    try:
        df, meta = get_prices(db, ticker, period="1y")
        # Si el cache no tiene suficiente histórico, fuerza descarga fresca de 1 año
        if df.empty or len(df) < sigma_window + 1:
            df, meta = get_prices(db, ticker, period="1y", fresh=True)
        if df.empty or len(df) < sigma_window + 1:
            raise HTTPException(
                404,
                f"Datos insuficientes para '{ticker}' (necesita >{sigma_window} cierres, hay {len(df)}).",
            )

        closes = df["Close"].dropna().tail(sigma_window + 1).values
        if len(closes) < 5:
            raise HTTPException(404, f"Cierres insuficientes para σ histórica.")
        S = float(closes[-1])
        log_rets = np.diff(np.log(closes))
        sigma_ann = float(np.std(log_rets, ddof=1) * np.sqrt(252))

        # Rf: prioriza FRED, fallback a yfinance ^IRX
        r = None
        if fred.is_available():
            rf = fred.get_rf_rate_3m(db)
            if rf and rf.get("value_decimal") is not None:
                r = rf["value_decimal"]
        if r is None:
            try:
                irx_hist = yf.Ticker("^IRX").history(period="5d")
                r = float(irx_hist["Close"].iloc[-1]) / 100 if not irx_hist.empty else 0.04
            except Exception:
                r = 0.04

        K = S * K_pct
        pricer = OptionPricer(S=S, K=K, T=T, r=r, sigma=sigma_ann)
        result = pricer.summary(option_type)  # type: ignore[arg-type]
        result["ticker"]            = ticker
        result["spot_used"]         = round(S, 6)
        result["strike_pct_of_spot"] = K_pct
        result["sigma_source"]      = f"historical_{sigma_window}d"
        result["rf_source"]         = "FRED.DGS3MO" if fred.is_available() else "yfinance.^IRX or default"
        result["price_cache_meta"]  = meta
        return json_response(result)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"Error valorando opción sobre {ticker}: {exc}")
    finally:
        db.close()


@app.post("/api/v1/opcion/precio", summary="Black-Scholes + Greeks + paridad put-call (parámetros libres) — M10")
def post_option_price(req: OptionRequest):
    """Valoración Black-Scholes para opción europea con sus Greeks."""
    from backend.app.services.options import OptionPricer
    try:
        pricer = OptionPricer(S=req.S, K=req.K, T=req.T, r=req.r, sigma=req.sigma)
        return json_response(pricer.summary(req.option_type))  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"Error valorando opción: {exc}")


class BondRequest(BaseModel):
    face_value:     float = Field(..., gt=0, description="Valor nominal del bono.")
    coupon_rate:    float = Field(..., ge=0, le=1, description="Tasa cupón anual decimal (0.05 = 5%).")
    maturity_years: float = Field(..., gt=0, le=100, description="Vencimiento en años.")
    yield_rate:     float = Field(..., ge=0, le=1, description="Yield anual decimal.")
    frequency:      int   = Field(2, description="Pagos por año: 1, 2, 4 o 12.")

    @field_validator("frequency")
    @classmethod
    def _valid_freq(cls, v: int) -> int:
        if v not in (1, 2, 4, 12):
            raise ValueError("frequency debe ser 1, 2, 4 o 12")
        return v


@app.post("/api/v1/bono/duracion", summary="Bono sintético: precio, duración, convexidad — M9")
def post_bond_duration(req: BondRequest):
    """Calcula precio, duración Macaulay/modificada, convexidad y sensibilidad ante shocks."""
    from backend.app.services.bond import Bond
    try:
        bond = Bond(
            face_value=req.face_value,
            coupon_rate=req.coupon_rate,
            maturity_years=req.maturity_years,
            yield_rate=req.yield_rate,
            frequency=req.frequency,
        )
        return json_response(bond.summary())
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"Error calculando bono: {exc}")


@app.get("/api/v1/precios/cache", summary="Estado del cache de precios en SQLAlchemy")
def get_prices_cache():
    """Reporta cuántos activos y precios están cacheados en SQLite."""
    from backend.app.database import SessionLocal
    from backend.app.services.price_service import cache_summary
    db = SessionLocal()
    try:
        return json_response(cache_summary(db))
    finally:
        db.close()


@app.get("/api/v1/precios/{ticker}", summary="Precios OHLCV con cache transparente en SQLite")
def get_prices_with_cache(
    ticker: str,
    period: str = Query("2y", description="Período si no se especifican fechas (yfinance format)."),
    fresh:  bool = Query(False, description="Forzar descarga aunque haya cache fresco."),
):
    """Devuelve precios OHLCV de yfinance con cache transparente.

    - Si el cache tiene datos del último día, retorna cache (`cache_status: hit`).
    - Si no, descarga de yfinance con reintentos y persiste (`cache_status: miss`).
    - Si yfinance falla y hay cache vencido, lo reutiliza (`cache_status: stale_used`).
    """
    from backend.app.database import SessionLocal
    from backend.app.services.price_service import get_prices
    db = SessionLocal()
    try:
        df, meta = get_prices(db, ticker, period=period, fresh=fresh)
        if df.empty:
            raise HTTPException(404, f"Sin datos para '{ticker}' (cache_status={meta['cache_status']}).")
        return json_response({
            "ticker":   ticker,
            "n_rows":   len(df),
            "meta":     meta,
            "data":     df.assign(Date=lambda d: d["Date"].astype(str)).to_dict(orient="records"),
        })
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"Error consultando precios: {exc}")
    finally:
        db.close()


@app.get("/api/v1/curva-rendimiento", summary="Curva de rendimiento Tesoro EE.UU. + Nelson-Siegel — M9")
def get_yield_curve():
    """Curva spot del Tesoro EE.UU. desde FRED + ajuste Nelson-Siegel.

    Si FRED_API_KEY no está configurada, usa una curva DEMO ilustrativa
    marcada como `source: fallback_demo`.
    """
    from backend.app.database import SessionLocal
    from backend.app.services.yield_curve import YieldCurve

    db = SessionLocal()
    try:
        yc = YieldCurve(db)
        return json_response(yc.to_response())
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"Error construyendo la curva: {exc}")
    finally:
        db.close()


@app.get("/api/v1/signals/{ticker}", summary="Señales técnicas automáticas — M7")
def get_asset_signals(
    ticker: str,
    dates: DateRangeDep,
    _cfg: ConfigDep,
    rsi_overbought: int = Query(70, ge=50, le=99,
        description="Umbral de sobrecompra RSI (default 70)."),
    rsi_oversold:   int = Query(30, ge=1,  le=50,
        description="Umbral de sobreventa RSI (default 30)."),
    bollinger_std:  float = Query(2.0, gt=0.0, le=5.0,
        description="Desviaciones estándar para Bandas de Bollinger (default 2.0)."),
    persist: bool = Query(True,
        description="Guardar las señales disparadas en signals_log."),
):
    """Genera señales técnicas y opcionalmente las persiste en signals_log.

    Evita duplicados: solo persiste señales que no estén ya registradas para
    el mismo ticker, regla y fecha de hoy.
    """
    try:
        from datetime import datetime as _dt
        from sqlalchemy import and_
        from backend.app.database import SessionLocal
        from backend.app.models.db_models import SignalLog

        records = _build_technical_records(ticker, **dates)
        if not records or len(records) < 2:
            raise HTTPException(404, f"Datos insuficientes para '{ticker}'.")

        last, prev = records[-1], records[-2]
        rsi_val   = last.get("RSI") or 50
        macd_h    = last.get("MACD_Hist") or 0
        close     = last.get("Close") or 0
        bb_up     = last.get("BB_Upper") or 0
        bb_low    = last.get("BB_Lower") or 0

        signals = generate_signals({
            "RSI":            rsi_val,
            "MACD_Hist":      macd_h,
            "MACD_Hist_Prev": prev.get("MACD_Hist") or 0,
            "Close":          close,
            "BB_Upper":       bb_up,
            "BB_Lower":       bb_low,
            "RSI_Overbought": rsi_overbought,
            "RSI_Oversold":   rsi_oversold,
        })

        # Reglas adicionales aplicadas con los umbrales configurables del usuario
        extra: list[dict] = []
        if rsi_val >= rsi_overbought:
            extra.append({
                "id": "RSI_OVERBOUGHT_USER", "type": "sell",
                "value": float(rsi_val),
                "msg": f"RSI {rsi_val:.1f} >= umbral configurado ({rsi_overbought})",
            })
        if rsi_val <= rsi_oversold:
            extra.append({
                "id": "RSI_OVERSOLD_USER", "type": "buy",
                "value": float(rsi_val),
                "msg": f"RSI {rsi_val:.1f} <= umbral configurado ({rsi_oversold})",
            })
        if bb_up and close >= bb_up:
            extra.append({
                "id": "BB_UPPER_BREAK", "type": "sell",
                "value": float(close),
                "msg": f"Precio {close:.2f} toca/excede banda superior ({bb_up:.2f})",
            })
        if bb_low and close <= bb_low:
            extra.append({
                "id": "BB_LOWER_BREAK", "type": "buy",
                "value": float(close),
                "msg": f"Precio {close:.2f} toca/cae bajo banda inferior ({bb_low:.2f})",
            })

        all_signals = signals + extra

        # Persistencia en signals_log evitando duplicados del mismo día
        persisted = 0
        if persist:
            db = SessionLocal()
            try:
                today = _dt.utcnow().date()
                for s in all_signals:
                    rule = s.get("id") or s.get("type") or "unknown"
                    exists = db.query(SignalLog).filter(
                        and_(
                            SignalLog.ticker == ticker,
                            SignalLog.rule   == rule,
                        )
                    ).filter(SignalLog.timestamp >= _dt(today.year, today.month, today.day)).first()
                    if exists:
                        continue
                    db.add(SignalLog(
                        ticker=ticker,
                        rule=rule,
                        value=float(s.get("value")) if s.get("value") is not None else None,
                        note=str(s.get("msg") or "")[:255],
                    ))
                    persisted += 1
                if persisted:
                    db.commit()
            finally:
                db.close()

        return {
            "ticker":        ticker,
            "signals":       all_signals,
            "thresholds":    {
                "rsi_overbought": rsi_overbought,
                "rsi_oversold":   rsi_oversold,
                "bollinger_std":  bollinger_std,
            },
            "persisted_count": persisted,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/signals/{ticker}/history", summary="Historial de señales persistidas — M7")
def get_signals_history(
    ticker: str,
    limit: int = Query(100, ge=1, le=1000, description="Máximo de registros a devolver."),
):
    """Devuelve las señales persistidas en signals_log para el ticker dado."""
    try:
        from backend.app.database import SessionLocal
        from backend.app.models.db_models import SignalLog

        db = SessionLocal()
        try:
            rows = (
                db.query(SignalLog)
                .filter(SignalLog.ticker == ticker)
                .order_by(SignalLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            history = [
                {
                    "id":        r.id,
                    "timestamp": r.timestamp.isoformat(timespec="seconds"),
                    "ticker":    r.ticker,
                    "rule":      r.rule,
                    "value":     r.value,
                    "note":      r.note,
                }
                for r in rows
            ]
            return {"ticker": ticker, "count": len(history), "history": history}
        finally:
            db.close()
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


# ═════════════════════════════════════════════════════════════════════════════
# ALIAS DE ENDPOINTS CORTOS — exigidos por la guía del profesor
# Re-exportan los handlers existentes bajo paths cortos sin romper compatibilidad
# ═════════════════════════════════════════════════════════════════════════════

from backend.app.routers import aliases as _aliases_router  # noqa: E402
app.include_router(_aliases_router.router)


# Cada alias delega al handler original. Dependencias se reinyectan al pasar
# por el alias para que FastAPI valide los query params igual que en el original.

@app.get("/precios/{ticker}", tags=["alias-corto (guía profesor)"])
def alias_precios(ticker: str,
                  period: str = Query("2y"),
                  fresh:  bool = Query(False)):
    return get_prices_with_cache(ticker=ticker, period=period, fresh=fresh)


@app.get("/rendimientos/{ticker}", tags=["alias-corto (guía profesor)"])
def alias_rendimientos(ticker: str, dates: DateRangeDep, _cfg: ConfigDep):
    return get_returns_analysis(ticker=ticker, dates=dates, _cfg=_cfg)


@app.get("/indicadores/{ticker}", tags=["alias-corto (guía profesor)"])
def alias_indicadores(
    ticker: str, dates: DateRangeDep, _cfg: ConfigDep,
    sma_short:  int   = Query(20, ge=2,  le=200),
    sma_long:   int   = Query(50, ge=5,  le=400),
    ema_window: int   = Query(20, ge=2,  le=200),
    rsi_window: int   = Query(14, ge=2,  le=100),
    bb_window:  int   = Query(20, ge=5,  le=200),
    bb_std:     float = Query(2.0, gt=0, le=5),
):
    return get_technical_analysis(
        ticker=ticker, dates=dates, _cfg=_cfg,
        sma_short=sma_short, sma_long=sma_long, ema_window=ema_window,
        rsi_window=rsi_window, bb_window=bb_window, bb_std=bb_std,
    )


@app.get("/volatilidad/{ticker}", tags=["alias-corto (guía profesor)"])
def alias_volatilidad(ticker: str, dates: DateRangeDep, _cfg: ConfigDep,
                      lambda_ewma: float = Query(0.94, gt=0.0, lt=1.0)):
    return get_volatility_analysis(ticker=ticker, dates=dates, _cfg=_cfg, lambda_ewma=lambda_ewma)


@app.get("/capm/{ticker}", tags=["alias-corto (guía profesor)"])
def alias_capm(ticker: str, dates: DateRangeDep, var_p: VaRParamsDep, config: ConfigDep):
    """Alias corto: extrae solo el bloque CAPM del endpoint risk."""
    full = get_risk_analysis(ticker=ticker, dates=dates, var_p=var_p, config=config)
    # `full` es un Response — devolver tal cual mantiene contrato
    return full


def _resolve_rf_from_fred_or_fallback(config) -> tuple[float, str]:
    """Devuelve (rf_anual_decimal, source_label).

    Estrategia:
    1. Si FRED está disponible, usa DGS3MO con cache transparente.
    2. Si FRED falla o no hay key, usa yfinance ^IRX.
    3. Si ambos fallan, cae al config.default_rf de .env.
    """
    from backend.app.database import SessionLocal
    from backend.app.services import fred_service as fred
    import yfinance as yf

    if fred.is_available():
        db = SessionLocal()
        try:
            rf = fred.get_rf_rate_3m(db)
            if rf and rf.get("value_decimal") is not None:
                return float(rf["value_decimal"]), f"FRED.DGS3MO ({rf.get('date','')})"
        finally:
            db.close()

    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        if not irx.empty:
            return float(irx["Close"].iloc[-1]) / 100, "yfinance.^IRX"
    except Exception:
        pass

    return float(config.default_rf), f"config.default_rf ({config.default_rf})"


@app.get("/capm", tags=["alias-corto (guía profesor)"],
         summary="CAPM consolidado: tabla resumen para todos los activos del portafolio — M4")
def get_capm_consolidated(dates: DateRangeDep, config: ConfigDep):
    """Tabla CAPM completa con Rf obtenida automáticamente desde FRED.

    Para cada activo del portafolio devuelve:
    - Beta (regresión MCO)
    - Alpha de Jensen
    - R²
    - Rendimiento esperado CAPM (anual)
    - Clasificación (agresivo / defensivo / neutro)
    - Descomposición de varianza (sistemática vs idiosincrática)
    - Tasa libre de riesgo usada y su fuente
    - Discusión de diversificación
    """
    rf_annual, rf_source = _resolve_rf_from_fred_or_fallback(config)

    bench_data = get_historical_data(config.benchmark, **dates)
    if bench_data is None:
        raise HTTPException(404, f"No hay datos para benchmark '{config.benchmark}'.")
    bench_ret, _ = calculate_returns(bench_data)

    market_mean_daily   = float(bench_ret.mean())
    market_mean_annual  = (1 + market_mean_daily) ** 252 - 1
    market_premium_ann  = market_mean_annual - rf_annual

    rows: list[dict] = []
    for ticker in config.tickers:
        data = get_historical_data(ticker, **dates)
        if data is None:
            rows.append({"ticker": ticker, "error": "sin datos"})
            continue
        ret, _ = calculate_returns(data)
        common = ret.index.intersection(bench_ret.index)
        if len(common) < 30:
            rows.append({"ticker": ticker, "error": f"insuficientes (n={len(common)})"})
            continue
        capm = calculate_capm(ret.loc[common], bench_ret.loc[common], rf_rate=rf_annual)
        vd = capm["Variance_Decomposition"]
        rows.append({
            "ticker":              ticker,
            "beta":                round(capm["Beta"], 4),
            "alpha_daily":         round(capm["Alpha"], 6),
            "r_squared":           round(capm["R_Squared"], 4),
            "expected_return_annual": round(capm["Expected_Return_Annual"], 4),
            "classification":      capm["Classification"],
            "classification_note": capm["Classification_Note"],
            "systematic_share":    round(vd["systematic_share"], 4) if vd["systematic_share"] is not None else None,
            "var_systematic":      round(vd["var_systematic"], 8),
            "var_idiosyncratic":   round(vd["var_idiosyncratic"], 8),
        })

    diversification_note = (
        "El riesgo total de un activo se descompone en dos partes: la sistemática (β²·σ²_m), "
        "que es la sensibilidad al mercado y NO se puede diversificar; y la idiosincrática (σ²_ε), "
        "específica del activo, que SÍ se reduce combinando activos descorrelacionados. "
        "A medida que el portafolio crece, la varianza idiosincrática tiende a cero (LGN), pero "
        "siempre queda el riesgo sistemático: ese es el límite teórico de la diversificación."
    )

    return json_response({
        "benchmark":              config.benchmark,
        "rf_annual_used":         round(rf_annual, 6),
        "rf_source":              rf_source,
        "market_mean_annual":     round(market_mean_annual, 6),
        "market_premium_annual":  round(market_premium_ann, 6),
        "n_assets_evaluated":     len([r for r in rows if "error" not in r]),
        "summary_table":          rows,
        "diversification_discussion": diversification_note,
    })


class VarRequest(BaseModel):
    tickers:        list[str] = Field(..., min_length=1, max_length=20)
    weights:        list[float] = Field(..., min_length=1, max_length=20)
    confidence:     float = Field(0.95, ge=0.80, le=0.99,
        description="Nivel principal de confianza (siempre se reporta también el 99%).")
    n_simulations:  int   = Field(10_000, ge=10_000, le=200_000,
        description="Iteraciones Monte Carlo (mínimo 10,000 según instructivo).")
    seed:           int   = Field(42, ge=0, le=2**31 - 1,
        description="Semilla NumPy para reproducibilidad del Monte Carlo.")
    lookback_days:  int   = Field(500, ge=250, le=2520,
        description="Ventana de backtesting Kupiec (mínimo 250 días según instructivo).")

    @model_validator(mode="after")
    def _weights_sum(self) -> "VarRequest":
        if len(self.tickers) != len(self.weights):
            raise ValueError("tickers y weights deben tener la misma longitud")
        s = sum(self.weights)
        if abs(s - 1.0) > 1e-3:
            raise ValueError(f"weights deben sumar 1; suman {s:.4f}")
        for w in self.weights:
            if not isinstance(w, (int, float)):
                raise ValueError("Todos los pesos deben ser numéricos")
        return self


def _build_var_block(portfolio_ret: pd.Series, confidence: float,
                     n_simulations: int, seed: int) -> dict:
    """Calcula VaR diario y anualizado para los 3 métodos en un nivel de confianza."""
    sqrt252 = math.sqrt(252)
    raw = calculate_var_cvar(portfolio_ret, confidence=confidence,
                             n_simulations=n_simulations, seed=seed)
    out: dict[str, dict] = {}
    for method in ("Historico", "Parametrico", "Montecarlo"):
        var_d  = raw[method]["VaR"]
        cvar_d = raw[method]["CVaR"]
        out[method] = {
            "VaR_daily":   round(var_d, 8),
            "VaR_annual":  round(var_d * sqrt252, 8),
            "CVaR_daily":  round(cvar_d, 8),
            "CVaR_annual": round(cvar_d * sqrt252, 8),
        }
    out["confidence"] = confidence
    if "distribution" in raw["Montecarlo"]:
        out["Montecarlo"]["distribution"] = raw["Montecarlo"]["distribution"]
        out["Montecarlo"]["seed"]         = raw["Montecarlo"]["seed"]
    return out


@app.post("/var", tags=["alias-corto (guía profesor)"],
          summary="VaR/CVaR del portafolio + Kupiec en 3 métodos — M5")
def alias_var(req: VarRequest, config: ConfigDep):
    """VaR paramétrico, histórico y Monte Carlo + CVaR + Kupiec POF en los 3 métodos.

    Devuelve siempre los niveles 95% Y 99%, valores diarios y anualizados,
    tabla comparativa con interpretación de diferencias, datos para histograma
    con líneas verticales VaR/CVaR, y backtesting Kupiec con verdict textual
    indicando si cada método subestima/sobreestima/es correcto.
    """
    all_rets: dict[str, pd.Series] = {}
    for t in req.tickers:
        data = get_historical_data(t, period="2y")
        if data is None:
            raise HTTPException(404, f"Sin datos para '{t}'.")
        ret, _ = calculate_returns(data)
        all_rets[t] = ret
    df = pd.DataFrame(all_rets).dropna()

    weights = np.array(req.weights)
    portfolio_ret = (df.values * weights).sum(axis=1)
    portfolio_ret = pd.Series(portfolio_ret).replace([np.inf, -np.inf], np.nan).dropna()

    # Recortar a la ventana de backtesting solicitada
    lookback = min(req.lookback_days, len(portfolio_ret))
    portfolio_lb = portfolio_ret.iloc[-lookback:]

    # ── Bloques 95% Y 99% ────────────────────────────────────────────────
    block_main = _build_var_block(portfolio_lb, req.confidence, req.n_simulations, req.seed)
    block_99   = _build_var_block(portfolio_lb, 0.99,            req.n_simulations, req.seed)

    # ── Kupiec en los 3 métodos al nivel principal ──────────────────────
    raw_main = calculate_var_cvar(portfolio_lb, confidence=req.confidence,
                                  n_simulations=req.n_simulations, seed=req.seed)
    kupiec: dict = {}
    for method in ("Historico", "Parametrico", "Montecarlo"):
        var_val = raw_main[method]["VaR"]
        kupiec[method] = kupiec_test(portfolio_lb, var_val, req.confidence)

    passes_summary = {m: kupiec[m]["passed"] for m in ("Historico", "Parametrico", "Montecarlo")}
    methods_pass = [m for m, p in passes_summary.items() if p is True]

    # ── Tabla comparativa con interpretación ────────────────────────────
    comparison_rows = []
    for method in ("Historico", "Parametrico", "Montecarlo"):
        d = block_main[method]
        comparison_rows.append({
            "method":      method,
            "VaR_daily":   d["VaR_daily"],
            "VaR_annual":  d["VaR_annual"],
            "CVaR_daily":  d["CVaR_daily"],
            "CVaR_annual": d["CVaR_annual"],
            "kupiec_passes": kupiec[method]["passed"],
            "kupiec_verdict": kupiec[method]["verdict"],
        })

    diff_interp_parts: list[str] = []
    var_h = block_main["Historico"]["VaR_daily"]
    var_p = block_main["Parametrico"]["VaR_daily"]
    var_m = block_main["Montecarlo"]["VaR_daily"]
    if var_h > var_p * 1.05:
        diff_interp_parts.append(
            f"Histórico ({var_h:.4f}) > Paramétrico ({var_p:.4f}) → la distribución empírica tiene colas más pesadas que la normal asumida.")
    if abs(var_m - var_p) / max(var_p, 1e-9) < 0.10:
        diff_interp_parts.append(
            "Monte Carlo ≈ Paramétrico → ambos asumen normalidad y producen valores similares.")
    if methods_pass:
        diff_interp_parts.append(f"Pasan Kupiec: {', '.join(methods_pass)}.")
    else:
        diff_interp_parts.append("Ningún método pasa Kupiec en esta ventana — revisar régimen del activo.")

    # ── Datos para histograma con líneas verticales VaR/CVaR ────────────
    hist_data = portfolio_lb.tolist()
    chart_data = {
        "portfolio_returns": _safe_json(hist_data[-500:]),  # cap para no sobrecargar
        "n_returns":         int(len(portfolio_lb)),
        "vertical_lines": {
            "VaR_Historico_95":   -block_main["Historico"]["VaR_daily"],
            "VaR_Parametrico_95": -block_main["Parametrico"]["VaR_daily"],
            "VaR_Montecarlo_95":  -block_main["Montecarlo"]["VaR_daily"],
            "CVaR_Historico_95":  -block_main["Historico"]["CVaR_daily"],
            "VaR_Historico_99":   -block_99["Historico"]["VaR_daily"],
        },
    }

    return json_response({
        "tickers":       req.tickers,
        "weights":       req.weights,
        "confidence":    req.confidence,
        "n_simulations": req.n_simulations,
        "seed":          req.seed,
        "lookback_days_used": lookback,
        "n_returns_total":    int(len(portfolio_ret)),
        # Bloques principales del response (conservados + nuevos)
        "var_cvar":      raw_main,                 # legacy
        "kupiec":        kupiec,                   # legacy
        # Nuevos campos del instructivo
        "parametric":    block_main["Parametrico"],
        "historical":    block_main["Historico"],
        "montecarlo":    block_main["Montecarlo"],
        "cvar":          {
            "Historico":   block_main["Historico"]["CVaR_daily"],
            "Parametrico": block_main["Parametrico"]["CVaR_daily"],
            "Montecarlo":  block_main["Montecarlo"]["CVaR_daily"],
            "interpretation": (
                "CVaR (Expected Shortfall) mide la pérdida promedio condicionada a "
                "que se exceda el VaR. Es siempre >= VaR y captura la severidad del riesgo de cola."
            ),
        },
        "var_99":        block_99,
        "comparison_table": {
            "rows":           comparison_rows,
            "interpretation": " ".join(diff_interp_parts),
        },
        "kupiec_test": {
            "lookback_days":  lookback,
            "confidence":     req.confidence,
            "by_method":      kupiec,
            "passes_summary": passes_summary,
            "methods_passing": methods_pass,
            "interpretation": (
                f"Kupiec POF aplicado a los 3 métodos en ventana de {lookback} días "
                f"(mínimo exigido: 250). Pasan: {methods_pass if methods_pass else 'ninguno'}. "
                "LR_POF se compara contra chi²(1)=3.84 al 95%."
            ),
        },
        "chart_data":    chart_data,
    })


@app.post("/frontera-eficiente", tags=["alias-corto (guía profesor)"])
def alias_frontera(dates: DateRangeDep, config: ConfigDep,
                   include_qp: bool = Query(True)):
    return get_portfolio_optimization(dates=dates, config=config, include_qp=include_qp)


@app.get("/alertas/{ticker}", tags=["alias-corto (guía profesor)"])
def alias_alertas(ticker: str, dates: DateRangeDep, _cfg: ConfigDep,
                  rsi_overbought: int = Query(70, ge=50, le=99),
                  rsi_oversold:   int = Query(30, ge=1, le=50),
                  bollinger_std:  float = Query(2.0, gt=0.0, le=5.0),
                  persist:        bool = Query(True)):
    return get_asset_signals(ticker=ticker, dates=dates, _cfg=_cfg,
                             rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold,
                             bollinger_std=bollinger_std, persist=persist)


@app.get("/macro", tags=["alias-corto (guía profesor)"])
def alias_macro():
    return get_macro_indicators(db=None)


@app.get("/curva-rendimiento", tags=["alias-corto (guía profesor)"])
def alias_curva():
    return get_yield_curve()


@app.post("/bono/duracion", tags=["alias-corto (guía profesor)"])
def alias_bono(req: BondRequest):
    return post_bond_duration(req=req)


@app.post("/opcion/precio", tags=["alias-corto (guía profesor)"])
def alias_opcion(req: OptionRequest):
    return post_option_price(req=req)


@app.post("/stress", tags=["alias-corto (guía profesor)"])
def alias_stress(req: StressRequest):
    return post_stress(req=req)


@app.post("/predict", tags=["alias-corto (guía profesor)"])
def alias_predict(req: PredictRequest):
    return post_predict(req=req)
