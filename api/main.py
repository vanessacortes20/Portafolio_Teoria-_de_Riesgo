"""
RiskLab USTA — FastAPI Backend
Pydantic validation + dependency injection + date-range & VaR parameters.
"""
from __future__ import annotations

import json
import math
import secrets
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, Optional

import bcrypt
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import re as _re
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from scipy import stats as scipy_stats
from starlette.responses import Response

from api.config import Settings, get_settings
from api.data import get_historical_data
from api.services import (
    Bond,
    DataService,
    FredClient,
    OptionPricer,
    StressTester,
    YieldCurve,
    get_data_service,
    get_fred_client,
)
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
    calculate_bollinger_bands,
    calculate_capm,
    calculate_ema,
    calculate_macd,
    calculate_returns,
    calculate_rsi,
    calculate_sma,
    calculate_stochastic,
    calculate_var_cvar,
    compare_markowitz_with_without_short,
    compute_efficient_frontier_qp,
    compute_ewma_comparison,
    compute_ewma_volatility,
    ewma_vs_garch_table,
    fit_garch_models,
    generate_signals,
    get_descriptive_stats,
    kupiec_test,
    optimize_portfolio,
    optimize_portfolio_target_return,
    perform_normality_tests,
    solve_markowitz_qp,
)

# ── Auth config ──────────────────────────────────────────────────────────────
_settings_boot = get_settings()
_SECRET_KEY = _settings_boot.jwt_secret
_ALGORITHM  = "HS256"
_TOKEN_TTL  = _settings_boot.jwt_ttl_minutes

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
    init_db()
    seed_demo_users(_hash)


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


# ─── Configuración inyectable ─────────────────────────────────────────────────
# Settings centraliza .env + variables de entorno (api/config.py).
# Se inyecta vía Depends en cada ruta que la necesite.
ConfigDep = Annotated[Settings, Depends(get_settings)]


# ─── Rango de fechas disponible ───────────────────────────────────────────────
DATA_MIN_DATE = date.fromisoformat(_settings_boot.data_min_date)
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


@app.get(
    "/api/v1/cache/stats",
    summary="Estado del cache de datos externos en SQLite",
)
def get_cache_stats(svc: DataService = Depends(get_data_service)):
    """
    Estadisticas del cache transparente que respalda los endpoints de datos.

    Retorna:
      - assets:           cantidad de tickers conocidos.
      - price_rows:       cantidad total de filas OHLCV persistidas.
      - last_fetched_at:  timestamp UTC de la ultima descarga.
      - ttl_hours:        tiempo de vida configurado para los datos.
    """
    return svc.cache_stats()


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


@app.get("/api/v1/volatility/{ticker}", summary="EWMA + ARCH/GARCH/EGARCH — M3")
def get_volatility_analysis(
    ticker: str,
    dates: DateRangeDep,
    _cfg: ConfigDep,
    ewma_lambda: float = Query(
        0.94,
        ge=0.50,
        le=0.999,
        description="Factor de decay para EWMA (RiskMetrics usa 0.94).",
    ),
    ewma_extra_lambdas: Optional[str] = Query(
        None,
        description="Lambdas adicionales a comparar (CSV, ej: '0.90,0.97').",
    ),
):
    try:
        data = get_historical_data(ticker, **dates)
        if data is None:
            raise HTTPException(404, f"No se encontraron datos para '{ticker}'.")
        _, log_ret = calculate_returns(data)

        result = fit_garch_models(log_ret)

        # ── Residuos estandarizados y pronóstico a 10 días ──────────────────
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

        # ── EWMA (RiskMetrics) ──────────────────────────────────────────────
        ewma_series = compute_ewma_volatility(log_ret, lambda_=ewma_lambda)
        if not ewma_series.empty:
            last_sigma_daily = float(ewma_series.iloc[-1])
            result["EWMA"] = {
                "lambda":                   ewma_lambda,
                "volatility":               _safe_json(ewma_series.tolist()),
                "current_volatility_daily": _sv(last_sigma_daily),
                "current_volatility_annual": _sv(last_sigma_daily * math.sqrt(252)),
            }
        else:
            result["EWMA"] = {
                "lambda":                   ewma_lambda,
                "volatility":               [],
                "current_volatility_daily": None,
                "current_volatility_annual": None,
            }

        # ── Comparación con λ adicionales ───────────────────────────────────
        if ewma_extra_lambdas:
            try:
                extras = [
                    float(x.strip())
                    for x in ewma_extra_lambdas.split(",")
                    if x.strip()
                ]
                extras = [x for x in extras if 0.5 < x < 0.999]
                if extras:
                    result["EWMA_Comparison"] = _safe_json(
                        compute_ewma_comparison(log_ret, extras)
                    )
            except ValueError:
                pass

        # ── Tabla comparativa cualitativa EWMA vs GARCH ─────────────────────
        result["EWMA_vs_GARCH_Table"] = ewma_vs_garch_table()

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


# ─── Markowitz formulado como QP explícito (M6) ───────────────────────────────


class FrontierRequest(BaseModel):
    """Parámetros para la frontera eficiente vía QP."""
    tickers:     Optional[list[str]] = Field(
        None, description="Lista de tickers; si se omite usa los del Settings."
    )
    allow_short: bool = Field(
        False, description="Permitir pesos negativos (ventas en corto)."
    )
    n_points:    int = Field(
        50, ge=10, le=200, description="Cantidad de puntos en la frontera."
    )
    rf_rate:     Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Tasa libre de riesgo anual; si se omite usa Settings.",
    )

    @field_validator("tickers")
    @classmethod
    def _validate_tickers(cls, v):
        if v is not None and len(v) < 2:
            raise ValueError("Se requieren al menos 2 tickers para optimizar.")
        return v


def _load_returns_df(
    tickers: list[str],
    dates: dict,
) -> pd.DataFrame:
    """Helper interno: descarga retornos diarios para una lista de tickers."""
    all_rets: dict[str, pd.Series] = {}
    for t in tickers:
        data = get_historical_data(t, **dates)
        if data is not None:
            ret, _ = calculate_returns(data)
            all_rets[t] = ret
    if len(all_rets) < 2:
        raise HTTPException(
            500, "No hay suficientes activos con datos disponibles."
        )
    return pd.DataFrame(all_rets).dropna()


@app.post(
    "/api/v1/frontier",
    summary="Frontera eficiente Markowitz vía QP (cvxpy) — M6",
)
def post_efficient_frontier(
    body: FrontierRequest,
    dates: DateRangeDep,
    config: ConfigDep,
):
    """
    Resuelve el problema de Markowitz como programación cuadrática explícita:

        min   wᵀ Σ w
        s.t.  Σ wᵢ = 1
              μᵀ w = μ*           (recorre n_points valores)
              wᵢ ≥ 0               (si allow_short=False)

    Retorna la frontera completa, el portafolio de mínima varianza y el
    portafolio de máximo Sharpe Ratio sobre la frontera.
    """
    try:
        tickers = body.tickers or config.tickers
        rf_rate = body.rf_rate if body.rf_rate is not None else config.default_rf
        df_rets = _load_returns_df(tickers, dates)
        result = compute_efficient_frontier_qp(
            df_rets,
            allow_short=body.allow_short,
            n_points=body.n_points,
            rf_rate=rf_rate,
        )
        return json_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get(
    "/api/v1/frontier/compare",
    summary="Comparativa Markowitz con y sin no-negatividad — M6",
)
def get_frontier_compare(
    dates: DateRangeDep,
    config: ConfigDep,
    n_points: int = Query(50, ge=10, le=200),
):
    """
    Resuelve Markowitz como QP en dos versiones (permitiendo y prohibiendo
    ventas en corto) y reporta el costo de imponer la restricción w ≥ 0:
    cuánto sube la volatilidad, cuánto baja el Sharpe, qué activos caen a
    peso cero en la versión restringida.
    """
    try:
        df_rets = _load_returns_df(config.tickers, dates)
        return json_response(
            compare_markowitz_with_without_short(
                df_rets, rf_rate=config.default_rf, n_points=n_points
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ─── Renta fija: curva, Nelson-Siegel, duración y convexidad (M9) ─────────────


@app.get(
    "/api/v1/yield-curve",
    summary="Curva de tesoros US con ajuste Nelson-Siegel — M9",
)
def get_yield_curve(
    fit_ns: bool = Query(True, description="Si True, ajusta Nelson-Siegel."),
    fred: FredClient = Depends(get_fred_client),
):
    """
    Construye la curva de rendimiento spot a partir de FRED (con fallback a
    yfinance si la API key no esta configurada). Plazos: 3M, 1Y, 2Y, 5Y,
    10Y y 30Y.

    Con fit_ns=True ajusta Nelson-Siegel sobre los puntos crudos y retorna
    los 4 parametros (beta0, beta1, beta2, lambda) + RMSE, mas la curva
    ajustada evaluada en una grilla densa.
    """
    try:
        raw = fred.get_yield_curve()
        points = raw["points"]
        if len(points) < 3:
            raise HTTPException(503, "Datos insuficientes para construir la curva.")

        maturities = list(points.keys())
        yields = [points[m]["yield"] for m in maturities]

        result: dict = {
            "as_of":   raw["as_of"],
            "source":  raw["source"],
            "points": [
                {
                    "maturity_years": m,
                    "yield":          points[m]["yield"],
                    "series_id":      points[m].get("series_id"),
                    "date":           points[m].get("date"),
                    "source":         points[m].get("source"),
                }
                for m in maturities
            ],
        }

        if fit_ns:
            curve = YieldCurve(maturities, yields)
            ns = curve.fit_nelson_siegel()
            grid = np.linspace(0.1, max(maturities) * 1.1, 60)
            fitted = [
                {"maturity_years": float(t), "yield": float(curve.spot_rate(float(t)))}
                for t in grid
            ]
            result["nelson_siegel"] = ns.to_dict()
            result["fitted_curve"] = fitted
            result["shape"] = curve.shape()

        return json_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


class BondRequest(BaseModel):
    """Especificacion de un bono sintetico para el calculo de duracion/convexidad."""
    face: float = Field(1000.0, gt=0, le=1_000_000_000, description="Valor nominal.")
    coupon_rate: float = Field(
        0.05, ge=0.0, le=1.0, description="Cupon anual (decimal)."
    )
    maturity_years: float = Field(
        10.0, gt=0, le=50, description="Vencimiento en anos."
    )
    freq: int = Field(2, description="Pagos por anio (1, 2 o 4).")
    ytm: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="YTM (decimal). Si se omite se toma de la curva NS al plazo del bono.",
    )

    @field_validator("freq")
    @classmethod
    def _validate_freq(cls, v: int) -> int:
        if v not in (1, 2, 4):
            raise ValueError("freq debe ser 1 (anual), 2 (semestral) o 4 (trimestral).")
        return v


@app.post(
    "/api/v1/bond/duration",
    summary="Duración, convexidad y sensibilidad de un bono sintético — M9",
)
def post_bond_duration(
    body: BondRequest,
    fred: FredClient = Depends(get_fred_client),
):
    """
    Calcula precio, duracion de Macaulay, duracion modificada y convexidad
    de un bono sintetico, mas la tabla de sensibilidad ante shocks de
    +-50, +-100 y +-200 puntos basicos comparando tres aproximaciones:
      1. Lineal (solo duracion).
      2. Lineal + termino de convexidad.
      3. Reprice exacto descontando flujos.
    """
    try:
        ytm = body.ytm
        if ytm is None:
            raw = fred.get_yield_curve()
            pts = raw["points"]
            if len(pts) < 3:
                raise HTTPException(
                    503,
                    "Sin datos de curva para inferir YTM. Especifique ytm en el body.",
                )
            maturities = list(pts.keys())
            yields = [pts[m]["yield"] for m in maturities]
            curve = YieldCurve(maturities, yields)
            ytm = float(curve.spot_rate(body.maturity_years))

        bond = Bond(
            face=body.face,
            coupon_rate=body.coupon_rate,
            maturity_years=body.maturity_years,
            freq=body.freq,
        )

        summary = bond.summary(ytm)
        sensitivity = bond.sensitivity_table(ytm)
        cash_flows = [
            {"t_years": t, "cash_flow": cf} for t, cf in bond.cash_flows()
        ]

        return json_response(
            {
                **summary,
                "ytm_source":  "curve_ns" if body.ytm is None else "user",
                "cash_flows":  cash_flows,
                "sensitivity": sensitivity,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ─── Opciones europeas Black-Scholes (M10) ───────────────────────────────────


class OptionRequest(BaseModel):
    """Parametros de una opcion europea para valoracion via Black-Scholes."""
    S: float = Field(..., gt=0, le=1e9, description="Precio actual del subyacente.")
    K: float = Field(..., gt=0, le=1e9, description="Strike (precio de ejercicio).")
    T: float = Field(..., gt=0, le=30, description="Tiempo a vencimiento en anos.")
    r: float = Field(..., ge=0, le=1, description="Tasa libre de riesgo anual (decimal).")
    sigma: float = Field(..., gt=0, le=5, description="Volatilidad anual (decimal).")
    option_type: Literal["call", "put"] = Field(..., description="Tipo de opcion.")
    market_price: Optional[float] = Field(
        None, gt=0, le=1e9,
        description="Precio observado de mercado. Si se incluye, calcula volatilidad implicita.",
    )

    @field_validator("sigma")
    @classmethod
    def _validate_sigma(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("sigma debe ser positivo.")
        return v


class GreeksModel(BaseModel):
    """Las cinco Greeks de Black-Scholes (raw, por unidad de cambio)."""
    delta: float
    gamma: float
    vega:  float
    theta: float
    rho:   float


class OptionResponse(BaseModel):
    """Respuesta tipada del endpoint de valoracion de opciones."""
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: Literal["call", "put"]
    price: float
    greeks: GreeksModel
    put_call_parity: dict
    implied_volatility: Optional[dict] = None


@app.post(
    "/api/v1/option/price",
    response_model=OptionResponse,
    summary="Black-Scholes: precio, Greeks, paridad put-call y σ implícita — M10",
)
def post_option_price(body: OptionRequest):
    """
    Valora una opcion europea por Black-Scholes:

        Call = S * N(d1) - K * exp(-rT) * N(d2)
        Put  = K * exp(-rT) * N(-d2) - S * N(-d1)

    Retorna las cinco Greeks (Delta, Gamma, Vega, Theta, Rho), verifica la
    paridad put-call sobre el par (C, P) y, si se incluye market_price,
    resuelve la volatilidad implicita por Newton-Raphson.
    """
    try:
        pricer = OptionPricer(S=body.S, K=body.K, T=body.T, r=body.r, sigma=body.sigma)
        price = pricer.price(body.option_type)
        greeks = pricer.greeks(body.option_type)
        parity = pricer.put_call_parity()

        iv: Optional[dict] = None
        if body.market_price is not None:
            iv = pricer.implied_volatility(body.market_price, body.option_type)

        return OptionResponse(
            S=body.S, K=body.K, T=body.T, r=body.r, sigma=body.sigma,
            option_type=body.option_type,
            price=price,
            greeks=GreeksModel(**greeks),
            put_call_parity=parity,
            implied_volatility=iv,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/api/v1/option/scenarios/{ticker}",
    summary="Grilla de opciones BS sobre un activo del portafolio — M10",
)
def get_option_scenarios(
    ticker: str,
    dates: DateRangeDep,
    config: ConfigDep,
    moneyness_pct: float = Query(
        0.10, ge=0.01, le=0.50,
        description="Rango ± alrededor del spot para construir strikes.",
    ),
):
    """
    Construye una grilla de opciones europeas usando como subyacente el
    ticker indicado del portafolio:
      - S  : ultimo cierre disponible (cache transparente).
      - σ  : estimacion EWMA(λ=0.94) sobre log-retornos del periodo.
      - r  : tasa libre de riesgo desde Settings.default_rf.
      - K  : grilla [S*(1-x), S, S*(1+x)] con x = moneyness_pct.
      - T  : 30, 60, 90 y 180 dias en anos calendario.

    Para cada combinacion calcula precio call/put, Greeks y verifica
    numericamente la paridad put-call.
    """
    try:
        data = get_historical_data(ticker, **dates)
        if data is None or data.empty:
            raise HTTPException(404, f"No se encontraron datos para '{ticker}'.")
        _, log_ret = calculate_returns(data)
        if len(log_ret) < 20:
            raise HTTPException(422, "Historial insuficiente para estimar σ.")
        ewma_series = compute_ewma_volatility(log_ret, lambda_=0.94)
        if ewma_series.empty:
            raise HTTPException(422, "No se pudo estimar σ via EWMA.")
        sigma_daily = float(ewma_series.iloc[-1])
        sigma_annual = sigma_daily * math.sqrt(252)
        spot = float(data["Close"].iloc[-1])
        r = float(config.default_rf)

        strikes = [
            spot * (1 - moneyness_pct),
            spot,
            spot * (1 + moneyness_pct),
        ]
        maturities_days = [30, 60, 90, 180]

        grid: list[dict] = []
        for tdays in maturities_days:
            T_years = tdays / 365.0
            for K in strikes:
                pricer = OptionPricer(S=spot, K=K, T=T_years, r=r, sigma=sigma_annual)
                grid.append({
                    "T_days":          tdays,
                    "K":               float(K),
                    "moneyness":       float((K - spot) / spot),
                    "call":            pricer.price("call"),
                    "put":             pricer.price("put"),
                    "greeks_call":     pricer.greeks("call"),
                    "greeks_put":      pricer.greeks("put"),
                    "put_call_parity": pricer.put_call_parity(),
                })

        return json_response({
            "ticker":       ticker,
            "spot":         spot,
            "sigma_annual": sigma_annual,
            "r":            r,
            "grid":         grid,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ─── Stress testing (M11) ────────────────────────────────────────────────────


class ScenarioSpec(BaseModel):
    """Especificacion de un escenario de stress."""
    name:            str   = Field(..., min_length=1, max_length=60)
    rate_shock_bp:   float = Field(
        0.0, description="Shock paralelo a Rf en puntos basicos (+200 = subida).",
    )
    market_drop_pct: float = Field(
        0.0, description="Variacion del benchmark (-0.20 = caida 20%).",
    )
    vol_multiplier:  float = Field(
        1.0, description="Factor multiplicativo de sigma (2.0 = doble).",
    )

    @field_validator("rate_shock_bp")
    @classmethod
    def _v_rate(cls, v: float) -> float:
        if not -1000.0 <= v <= 1000.0:
            raise ValueError("rate_shock_bp fuera de rango (-1000 a +1000 pb).")
        return v

    @field_validator("market_drop_pct")
    @classmethod
    def _v_mkt(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError("market_drop_pct fuera de rango (-1.0 a +1.0).")
        return v

    @field_validator("vol_multiplier")
    @classmethod
    def _v_vol(cls, v: float) -> float:
        if not 0.0 < v <= 10.0:
            raise ValueError("vol_multiplier debe ser > 0 y <= 10.")
        return v


class StressRequest(BaseModel):
    """Body del endpoint POST /api/v1/stress."""
    tickers: Optional[list[str]] = Field(
        None, description="Lista de tickers; si se omite usa Settings.tickers."
    )
    weights: Optional[dict[str, float]] = Field(
        None,
        description="Pesos por ticker. Si se omite, usa el Max Sharpe del QP.",
    )
    scenarios: Optional[list[ScenarioSpec]] = Field(
        None,
        description="Lista de escenarios; si se omite usa los 6 por defecto.",
    )
    portfolio_value: float = Field(
        100_000.0, gt=0, le=1e12, description="Valor monetario del portafolio."
    )
    confidence: float = Field(
        0.95, ge=0.80, le=0.99, description="Nivel de confianza para el VaR."
    )
    equity_rate_duration: float = Field(
        0.0,
        ge=0.0,
        le=20.0,
        description=(
            "Sensibilidad implicita a la tasa para equities (0 = sin impacto "
            "directo del shock de tasa sobre el precio del activo)."
        ),
    )

    @field_validator("weights")
    @classmethod
    def _v_weights(cls, v: Optional[dict[str, float]]) -> Optional[dict[str, float]]:
        if v is None:
            return v
        if not v:
            raise ValueError("Si se especifica weights, no puede estar vacio.")
        s = sum(v.values())
        if abs(s - 1.0) > 0.05:
            raise ValueError(f"Los pesos deben sumar ~1.0 (suma actual: {s:.4f}).")
        return v


@app.post(
    "/api/v1/stress",
    summary="Stress testing del portafolio bajo escenarios extremos — M11",
)
def post_stress(
    body: StressRequest,
    dates: DateRangeDep,
    config: ConfigDep,
):
    """
    Aplica escenarios de stress al portafolio. Si no se pasan weights, el
    endpoint los calcula resolviendo el QP de Markowitz (Max Sharpe sin
    ventas en corto). Si no se pasan scenarios, ejecuta los 6 default:

      1. Tasa +200 pb        4. Mercado -30%
      2. Tasa -200 pb        5. Volatilidad x2
      3. Mercado -20%        6. Tormenta perfecta (tasa +200 + mkt -20% + sigma x2)

    Para cada escenario retorna:
      - perdida puntual del portafolio (% y monetaria),
      - VaR base vs VaR estresado al nivel de confianza solicitado y al 99%,
      - heatmap de impacto por activo (componente mercado, componente tasa,
        total, beta, peso).

    Tambien devuelve loss_bar (lista lista para graficar barras) y heatmap
    (estructura activo -> escenario -> impacto %).
    """
    try:
        tickers = body.tickers or config.tickers
        if len(tickers) < 2:
            raise HTTPException(422, "Se requieren al menos 2 tickers.")

        # ── Cargar retornos del universo ────────────────────────────────────
        df_rets = _load_returns_df(tickers, dates)

        # ── Cargar retornos del benchmark ───────────────────────────────────
        bench_data = get_historical_data(config.benchmark, **dates)
        if bench_data is None or bench_data.empty:
            raise HTTPException(503, "No se pudieron descargar datos del benchmark.")
        bench_ret, _ = calculate_returns(bench_data)

        # ── Resolver pesos: Max Sharpe del QP si el usuario no los pasa ─────
        if body.weights is None:
            qp_res = compute_efficient_frontier_qp(
                df_rets,
                allow_short=False,
                n_points=30,
                rf_rate=config.default_rf,
            )
            if not qp_res.get("feasible"):
                raise HTTPException(
                    500, "No se pudo resolver el portafolio Max Sharpe por defecto."
                )
            weights = qp_res["Max_Sharpe"]["Weights"]
        else:
            weights = dict(body.weights)

        # ── Construir tester y ejecutar ─────────────────────────────────────
        scenarios = (
            [s.model_dump() for s in body.scenarios] if body.scenarios else None
        )
        tester = StressTester(
            returns_df=df_rets,
            weights=weights,
            benchmark_returns=bench_ret,
            rf_rate=config.default_rf,
            portfolio_value=body.portfolio_value,
            equity_rate_duration=body.equity_rate_duration,
        )
        result = tester.run(scenarios=scenarios, confidence=body.confidence)
        return json_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/macro", summary="Indicadores macroeconómicos de contexto — M8")
def get_macro_indicators():
    """Devuelve tasa libre de riesgo, rendimiento del Tesoro a 10 años y retorno YTD del S&P 500."""
    import yfinance as yf
    result: dict = {"as_of": date.today().isoformat()}

    try:
        irx_hist = yf.Ticker("^IRX").history(period="5d")
        result["rf_rate"] = float(irx_hist["Close"].iloc[-1]) / 100 if not irx_hist.empty else 0.04
        result["rf_source"] = "^IRX T-Bill 13 sem."
    except Exception:
        result["rf_rate"] = 0.04
        result["rf_source"] = "default"

    try:
        tnx_hist = yf.Ticker("^TNX").history(period="5d")
        result["treasury_10y"] = float(tnx_hist["Close"].iloc[-1]) / 100 if not tnx_hist.empty else None
    except Exception:
        result["treasury_10y"] = None

    try:
        spy_hist = yf.Ticker("^GSPC").history(period="ytd")
        if not spy_hist.empty and len(spy_hist) > 1:
            result["spx_ytd"] = float(spy_hist["Close"].iloc[-1] / spy_hist["Close"].iloc[0] - 1)
        else:
            result["spx_ytd"] = None
    except Exception:
        result["spx_ytd"] = None

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
