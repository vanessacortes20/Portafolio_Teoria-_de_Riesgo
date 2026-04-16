"""
RiskLab USTA — Streamlit Dashboard v2.0
Mejoras: sidebar colapsable, interpretaciones dinámicas, selección de acciones,
tipografía mejorada, tablas opcionales, KPIs rediseñados, validadores de fechas y VaR.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

# ─── Config ──────────────────────────────────────────────────────────────────
load_dotenv()
TICKERS  = [t.strip() for t in os.getenv("PORTFOLIO_TICKERS", "NU,MELI,SONY,XOM,WPM").split(",")]
API_URL  = "http://localhost:8001/api/v1"

import streamlit as st
st.set_page_config(page_title="RiskLab USTA", layout="wide", page_icon="📈")

# ─── Paleta coherente ─────────────────────────────────────────────────────────
C = {
    "sidebar":   "#4b164c",
    "bg":        "#f5f7f9",
    "pink":      "#dd5e89",
    "orange":    "#f7a35c",
    "text":      "#1f2937",
    "white":     "#ffffff",
    "border":    "#e5e7eb",
    "green":     "#10b981",
    "red":       "#ef4444",
    "blue":      "#4e8cde",
    "yellow":    "#f59e0b",
    "purple":    "#9b7fe8",
    "muted":     "#6b7280",
}

# Paleta unificada por activo (familia coherente, distinguible)
ASSET_COLORS: dict[str, str] = {
    "NU":   "#dd5e89",   # pink  — brand
    "MELI": "#f7a35c",   # coral orange
    "SONY": "#4e8cde",   # steel blue
    "XOM":  "#52b788",   # sage green
    "WPM":  "#9b7fe8",   # lavender
}

def asset_color(ticker: str) -> str:
    """Returns the brand color for a ticker, or a default if not in palette."""
    defaults = ["#dd5e89", "#f7a35c", "#4e8cde", "#52b788", "#9b7fe8",
                "#f59e0b", "#ef4444", "#10b981"]
    return ASSET_COLORS.get(ticker, defaults[TICKERS.index(ticker) % len(defaults)] if ticker in TICKERS else defaults[0])

_LEGEND = dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
               font=dict(color=C["text"], size=12))

PLOTLY_BASE = dict(
    font=dict(family="'Inter', 'Outfit', sans-serif", color=C["text"], size=13),
    paper_bgcolor=C["white"],
    plot_bgcolor="#fafafa",
    xaxis=dict(gridcolor="#f0f0f0", linecolor=C["border"], zeroline=False),
    yaxis=dict(gridcolor="#f0f0f0", linecolor=C["border"], zeroline=False),
    margin=dict(t=56, b=44, l=52, r=24),
    annotationdefaults=dict(font=dict(color=C["text"], size=12)),
    hoverlabel=dict(bgcolor=C["white"], bordercolor=C["border"],
                    font=dict(color=C["text"], size=12)),
)

def apply_style(fig, height=420):
    fig.update_layout(**PLOTLY_BASE, height=height, legend=_LEGEND)
    return fig


# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;600;700&display=swap');

/* ── Base ── */
.stApp {{
    background-color: {C["bg"]};
    font-family: 'Inter', 'Outfit', sans-serif;
    color: {C["text"]};
}}

/* ── Tipografía ── */
h1, h2, h3, h4, h5 {{ font-family: 'Outfit', sans-serif; color: {C["text"]}; }}

.rl-h1 {{
    font-family: 'Outfit', sans-serif;
    font-size: 2rem; font-weight: 700; line-height: 1.2;
    background: linear-gradient(90deg, {C["pink"]} 0%, {C["orange"]} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}}
.rl-sub {{
    color: {C["muted"]}; font-size: 0.95rem;
    font-weight: 400; margin-bottom: 1.5rem;
}}
.rl-section {{
    font-size: 1.05rem; font-weight: 600;
    color: {C["text"]}; margin: 1.5rem 0 0.6rem;
    border-left: 3px solid {C["pink"]};
    padding-left: 10px;
}}

/* ── Sidebar collapse button styling ── */
[data-testid="collapsedControl"] {{
    background: linear-gradient(135deg, {C["pink"]} 0%, {C["orange"]} 100%) !important;
    border-radius: 0 14px 14px 0 !important;
    box-shadow: 3px 0 16px rgba(221,94,137,0.35) !important;
    transition: box-shadow 0.25s ease, transform 0.2s ease !important;
    width: 26px !important;
    border: none !important;
}}
[data-testid="collapsedControl"]:hover {{
    box-shadow: 4px 0 22px rgba(221,94,137,0.55) !important;
    transform: scaleX(1.12) !important;
}}
[data-testid="collapsedControl"] svg {{
    color: white !important; stroke: white !important; fill: white !important;
}}
section[data-testid="stSidebar"] {{
    transition: width 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C["sidebar"]} 0%, #2d0a2e 100%);
    border-right: none;
}}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {{
    color: #ffffff !important;
}}
[data-testid="stSidebar"] .stButton>button {{
    width: 100%;
    background-color: rgba(255,255,255,0.07);
    color: white !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 10px;
    text-align: left; padding: 9px 14px; margin-bottom: 4px;
    font-size: 0.88rem; font-weight: 500;
    transition: all 0.18s;
}}
[data-testid="stSidebar"] .stButton>button:hover {{
    background-color: {C["pink"]} !important;
    border-color: {C["pink"]} !important;
    transform: translateX(4px);
}}
[data-testid="stSidebar"] .stDateInput label p,
[data-testid="stSidebar"] .stSelectbox label p,
[data-testid="stSidebar"] .stSelectbox label {{
    color: rgba(255,255,255,0.8) !important; font-size: 0.8rem !important;
}}
div[data-baseweb="select"] > div {{
    background-color: white !important; color: {C["text"]} !important;
}}

/* ── Métricas ── */
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {{
    color: {C["muted"]} !important; font-weight: 600; font-size: 0.82rem !important;
    text-transform: uppercase; letter-spacing: 0.04em;
}}
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] {{
    color: {C["text"]} !important; font-weight: 700; font-size: 1.6rem !important;
}}
[data-testid="stMetricDelta"] > div {{ color: {C["muted"]} !important; }}
[data-testid="metric-container"] {{
    background: {C["white"]}; border-radius: 14px;
    border: 1px solid {C["border"]};
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    padding: 16px 18px !important;
}}

/* ── KPI card ── */
.kpi-card {{
    background: {C["white"]}; border-radius: 14px;
    border: 1px solid {C["border"]};
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    padding: 18px 22px; margin-bottom: 14px;
}}
.kpi-label {{
    font-size: 0.75rem; font-weight: 600; color: {C["muted"]};
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
}}
.kpi-value {{
    font-size: 1.9rem; font-weight: 700; line-height: 1; color: {C["pink"]};
}}
.kpi-unit {{ font-size: 0.85rem; color: {C["muted"]}; margin-left: 3px; }}
.kpi-desc {{
    font-size: 0.78rem; color: {C["muted"]}; margin-top: 6px; line-height: 1.4;
}}

/* ── Section header ── */
.section-header {{
    font-size: 1.05rem; font-weight: 600; color: {C["text"]};
    border-left: 3px solid {C["pink"]}; padding-left: 10px;
    margin: 1.4rem 0 0.7rem;
}}

/* ── Info / interpretation box ── */
.interp-box {{
    background: #fdf4ff; border-left: 4px solid {C["pink"]};
    border-radius: 0 10px 10px 0; padding: 12px 16px;
    font-size: 0.88rem; color: {C["text"]}; margin: 10px 0;
    line-height: 1.55;
}}
.interp-box b {{ color: {C["sidebar"]}; }}
.interp-green {{
    background: #f0fdf4; border-left-color: {C["green"]};
}}
.interp-red {{ background: #fff5f5; border-left-color: {C["red"]}; }}
.interp-blue {{ background: #eff6ff; border-left-color: {C["blue"]}; }}

/* ── Signals ── */
.sig-buy {{
    background: #d1fae5; border: 1px solid #6ee7b7;
    border-left: 4px solid {C["green"]}; border-radius: 10px;
    padding: 12px 16px; margin: 8px 0; color: #065f46; font-weight: 600;
}}
.sig-sell {{
    background: #fee2e2; border: 1px solid #fca5a5;
    border-left: 4px solid {C["red"]}; border-radius: 10px;
    padding: 12px 16px; margin: 8px 0; color: #7f1d1d; font-weight: 600;
}}
.sig-explanation {{
    font-size: 0.82rem; font-weight: 400; margin-top: 5px;
    color: inherit; opacity: 0.8; line-height: 1.5;
}}
.sig-none {{
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid {C["green"]}; border-radius: 10px;
    padding: 20px; margin: 8px 0; text-align: center; color: #166534;
}}

/* ── Stat row ── */
.stat-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid {C["border"]};
    font-size: 0.9rem;
}}
.stat-row:last-child {{ border-bottom: none; }}
.stat-label {{ color: {C["muted"]}; font-weight: 400; }}
.stat-value {{ font-weight: 600; color: {C["text"]}; }}

/* ── Badges ── */
.badge-ok  {{ background:#d1fae5;color:#065f46;padding:3px 10px;border-radius:20px;font-size:0.78rem;font-weight:600; }}
.badge-err {{ background:#fee2e2;color:#7f1d1d;padding:3px 10px;border-radius:20px;font-size:0.78rem;font-weight:600; }}

/* ── Módulo tiles (Home) ── */
.mod-tile {{
    background: {C["white"]}; padding: 20px 16px; border-radius: 16px;
    text-align: center; border: 1px solid {C["border"]};
    height: 136px; display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.2s;
}}
.mod-tile:hover {{
    box-shadow: 0 6px 20px rgba(221,94,137,0.18);
    transform: translateY(-2px);
}}
.mod-icon {{ font-size: 1.9rem; margin-bottom: 6px; }}
.mod-id {{ font-size: 0.68rem; font-weight: 700; color:{C["pink"]};
           text-transform: uppercase; letter-spacing: 0.1em; }}
.mod-name {{ font-size: 0.82rem; font-weight: 600; color:{C["text"]}; }}

/* ── Main buttons ── */
.main .stButton>button {{
    background: linear-gradient(90deg, {C["pink"]}, {C["orange"]}) !important;
    color: white !important; border: none !important;
    border-radius: 10px; font-weight: 600;
    padding: 9px 16px; transition: opacity 0.18s;
}}
.main .stButton>button:hover {{ opacity: 0.85; }}

/* ── Future-work cards ── */
.fw-card {{
    background: {C["white"]}; border-radius: 14px;
    border: 1px solid {C["border"]};
    border-top: 3px solid {C["pink"]};
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    padding: 18px 20px; margin-bottom: 14px;
    transition: box-shadow 0.2s;
}}
.fw-card:hover {{ box-shadow: 0 6px 20px rgba(0,0,0,0.09); }}
.fw-card h4 {{ margin: 0 0 6px; font-size: 0.95rem; color:{C["text"]}; }}
.fw-card p  {{ margin: 0; font-size: 0.83rem; color:{C["muted"]}; line-height:1.5; }}
.fw-tag {{
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 2px 8px; border-radius: 20px;
    background: #fce7f3; color: #9d174d; margin-bottom: 8px;
}}

/* ── Date validation ── */
.date-ok  {{ color: {C["green"]}; font-size: 0.8rem; }}
.date-err {{ color: {C["red"]};   font-size: 0.8rem; }}

/* ── Divider / footer ── */
hr {{ border-color: {C["border"]}; margin: 1rem 0; }}
.footer {{ text-align:center; color:{C["muted"]}; font-size:0.82rem; padding:16px 0 6px; }}
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
_DEFAULTS = {
    "page": "Home",
    "date_start": date.today() - timedelta(days=730),
    "date_end": date.today(),
    "date_valid": True,
    "var_confidence": 0.95,
    "var_iterations": 10_000,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Utilidades ──────────────────────────────────────────────────────────────

def set_page(name: str):
    st.session_state.page = name
    st.rerun()


def fetch_api(endpoint: str, timeout: int = 20, extra_params: str = "") -> dict | list | None:
    try:
        url = f"{API_URL}{endpoint}"
        if extra_params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{extra_params}"
        r = requests.get(url, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def date_params() -> str:
    """Returns URL query string for the selected date range."""
    if not st.session_state.date_valid:
        return ""
    s = st.session_state.date_start
    e = st.session_state.date_end
    return f"start_date={s}&end_date={e}"


def render_api_error():
    st.markdown("""
        <div class='kpi-card' style='border-left:4px solid #ef4444;text-align:center;padding:30px'>
            <div style='font-size:2.4rem'>🚨</div>
            <h3 style='color:#ef4444;margin:10px 0 6px'>Sin conexión con la API</h3>
            <p style='color:#6b7280;margin:0 0 10px'>
                Asegúrate de que el servidor esté corriendo en el puerto 8001.
            </p>
            <code style='background:#f3f4f6;padding:4px 10px;border-radius:6px;font-size:0.83rem'>
                uvicorn api.main:app --reload --port 8001
            </code>
        </div>
    """, unsafe_allow_html=True)


def kpi(label: str, value: str, unit: str = "", color: str = None, desc: str = "") -> str:
    col = color or C["pink"]
    return f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value' style='color:{col}'>{value}<span class='kpi-unit'>{unit}</span></div>
        {"<div class='kpi-desc'>"+desc+"</div>" if desc else ""}
    </div>"""


def interp(text: str, style: str = "") -> str:
    cls = f"interp-box {style}" if style else "interp-box"
    return f"<div class='{cls}'>{text}</div>"


def norm_badge(p_value: float) -> str:
    if p_value > 0.05:
        return "<span class='badge-ok'>✔ Normal  (p > 0.05)</span>"
    return "<span class='badge-err'>✖ No Normal  (p ≤ 0.05)</span>"


# ─── Dynamic interpretation helpers ──────────────────────────────────────────

def interp_rsi(rsi: float, ticker: str) -> str:
    if rsi > 80:
        return (f"El RSI de <b>{rsi:.0f}</b> indica sobrecompra <b>extrema</b> en {ticker}. "
                "El activo ha subido muy rápido; históricamente esto coincide con pausas o correcciones.")
    if rsi > 70:
        return (f"Con un RSI de <b>{rsi:.0f}</b>, {ticker} está en zona de sobrecompra. "
                "No garantiza una caída inmediata, pero sugiere que el precio puede estar sobreextendido.")
    if rsi < 20:
        return (f"RSI de <b>{rsi:.0f}</b> — sobreventa extrema en {ticker}. "
                "El mercado está siendo muy pesimista; históricamente puede anticipar un rebote.")
    if rsi < 30:
        return (f"RSI en <b>{rsi:.0f}</b>: {ticker} está en zona de sobreventa. "
                "El activo ha caído rápidamente, lo que puede representar una oportunidad para inversores pacientes.")
    return (f"RSI de <b>{rsi:.0f}</b> en zona neutral para {ticker}. "
            "No hay señales extremas de sobrecompra ni sobreventa en este momento.")


def interp_volatility(vol_mean: float, ticker: str, best_model: str) -> str:
    vol_ann = vol_mean * (252 ** 0.5)
    if vol_ann > 0.50:
        level = "muy alta"
    elif vol_ann > 0.30:
        level = "alta"
    elif vol_ann > 0.15:
        level = "moderada"
    else:
        level = "baja"
    return (f"La volatilidad promedio de {ticker} es <b>{vol_mean:.2%} diaria "
            f"({vol_ann:.1%} anualizada)</b> — nivel {level}. "
            f"El modelo con mejor ajuste según AIC/BIC es <b>{best_model}</b>. "
            "Los picos en la gráfica muestran episodios de mayor incertidumbre (agrupamiento de volatilidad).")


def interp_capm(beta: float, alpha: float, r2: float, cls: str, ticker: str) -> str:
    beta_desc = (
        "amplifica los movimientos del mercado — cuando el S&P 500 sube 1%, este activo tiende a subir más"
        if beta > 1.2 else
        "amortigua los movimientos — reacciona menos que el mercado general"
        if beta < 0.8 else
        "se mueve aproximadamente en línea con el mercado"
    )
    alpha_note = (
        f"El Alpha positivo ({alpha:.4f}) sugiere que el activo ha generado retorno por encima de lo que predice el CAPM."
        if alpha > 0 else
        f"El Alpha negativo ({alpha:.4f}) indica que el activo ha rendido por debajo de lo esperado según su riesgo sistemático."
    )
    return (f"{ticker} tiene una <b>Beta de {beta:.3f}</b> — clasificado como <b>{cls}</b>. "
            f"Esto significa que el activo {beta_desc}. "
            f"El R² de {r2:.2%} indica qué tan bien explica el mercado los movimientos de este activo. "
            f"{alpha_note}")


def interp_var(var_hist: float, cvar_hist: float, confidence: float, ticker: str) -> str:
    pct = int(confidence * 100)
    return (f"Con un nivel de confianza del <b>{pct}%</b>, "
            f"la pérdida diaria máxima esperada de {ticker} es <b>{var_hist:.2%}</b> (VaR Histórico). "
            f"En los peores escenarios (el {100-pct}% de días más adversos), "
            f"la pérdida promedio es de <b>{cvar_hist:.2%}</b> (CVaR). "
            "Cuanto mayor es la diferencia entre VaR y CVaR, más pesadas son las colas de la distribución.")


def interp_returns(skew: float, kurt: float, ticker: str) -> str:
    skew_note = (
        "cola derecha larga (ganancias extremas ocasionales más probables que pérdidas extremas)"
        if skew > 0.3 else
        "cola izquierda larga (pérdidas extremas ocasionales más probables que ganancias extremas)"
        if skew < -0.3 else
        "distribución aproximadamente simétrica"
    )
    kurt_note = (
        f"la curtosis de {kurt:.2f} confirma colas pesadas (fat tails) — los eventos extremos ocurren con más frecuencia de lo que predice una distribución normal"
        if kurt > 3 else
        f"curtosis de {kurt:.2f} indica colas más ligeras que la normal"
    )
    return (f"Los rendimientos de {ticker} muestran <b>asimetría de {skew:.3f}</b> ({skew_note}), "
            f"y {kurt_note}. Esto es importante para el cálculo de riesgo: "
            "los modelos que asumen normalidad pueden subestimar eventos extremos.")


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2621/2621815.png", width=50)
    st.markdown("## RiskLab USTA")
    st.caption("Plataforma de análisis cuantitativo de riesgo financiero")
    st.markdown("---")

    ticker_selected = st.selectbox("Activo a analizar", TICKERS, key="ticker_sel")
    st.markdown(f"<p style='color:rgba(255,255,255,0.7);font-size:0.8rem'>Analizando: <b style='color:#f7a35c'>{ticker_selected}</b></p>",
                unsafe_allow_html=True)

    st.markdown("---")
    # ── Período de análisis con validación ──
    st.markdown("<p style='font-size:0.78rem;opacity:0.7;letter-spacing:0.08em;text-transform:uppercase'>📅 Período de Análisis</p>",
                unsafe_allow_html=True)
    today = date.today()
    d_start = st.date_input(
        "Desde", value=st.session_state.date_start,
        min_value=date(2010, 1, 1), max_value=today - timedelta(days=30),
        key="d_start_input",
    )
    d_end = st.date_input(
        "Hasta", value=st.session_state.date_end,
        min_value=date(2010, 1, 1), max_value=today,
        key="d_end_input",
    )

    date_ok = True
    if d_end <= d_start:
        st.error("⚠️ La fecha final debe ser posterior a la inicial.")
        date_ok = False
    elif (d_end - d_start).days < 90:
        st.warning("Mínimo 90 días para análisis estadísticamente confiable.")
        date_ok = False
    else:
        days = (d_end - d_start).days
        st.markdown(f"<p class='date-ok'>✔ {days} días seleccionados</p>", unsafe_allow_html=True)

    st.session_state.date_start  = d_start
    st.session_state.date_end    = d_end
    st.session_state.date_valid  = date_ok

    st.markdown("---")
    st.markdown("<p style='font-size:0.78rem;opacity:0.7;letter-spacing:0.08em;text-transform:uppercase'>Navegación</p>",
                unsafe_allow_html=True)
    if st.button("🏠 Inicio"):                         set_page("Home")
    if st.button("📊 Taller Gauss (M1–M4)"):           set_page("Taller Gauss")
    if st.button("🛡️ Gestión de Riesgo (M5–M8)"):     set_page("Gestion Riesgo")

    st.markdown("---")
    st.markdown("<p style='font-size:0.78rem;opacity:0.7;letter-spacing:0.08em;text-transform:uppercase'>Módulos</p>",
                unsafe_allow_html=True)
    for icon, label, target in [
        ("📈", "Análisis Técnico",    "1. Análisis Técnico"),
        ("📉", "Rendimientos",         "2. Rendimientos"),
        ("🌊", "Volatilidad GARCH",    "3. Volatilidad (GARCH)"),
        ("🔭", "CAPM & Beta",          "4. Riesgo y CAPM"),
        ("🛡️", "VaR / CVaR",          "5. Valor en Riesgo (VaR)"),
        ("💎", "Optimización",         "6. Optimización"),
        ("🔔", "Señales y Alertas",    "7. Señales y Alertas"),
        ("🌍", "Macroeconomía",        "8. Macro y Benchmark"),
        ("🚀", "Oportunidades",        "9. Oportunidades de Mejora"),
    ]:
        if st.button(f"{icon} {label}", key=f"nav_{target}"):
            set_page(target)

    st.markdown("---")
    st.markdown("<p style='font-size:0.72rem;opacity:0.5;text-align:center'>Usa el botón ◀ para ocultar este panel</p>",
                unsafe_allow_html=True)


# ─── Router ───────────────────────────────────────────────────────────────────
page = st.session_state.page
dp   = date_params()   # URL params string for date range

# ===========================================================================
#  HOME
# ===========================================================================
if page == "Home":
    st.markdown("<div class='rl-h1'>Dashboard de Inteligencia RiskLab</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>Selecciona un módulo para comenzar. Usa la barra lateral para filtrar por activo y período.</div>",
                unsafe_allow_html=True)

    tiles = [
        ("📈","M1","Análisis Técnico",  "1. Análisis Técnico"),
        ("📉","M2","Rendimientos",       "2. Rendimientos"),
        ("🌊","M3","Volatilidad GARCH",  "3. Volatilidad (GARCH)"),
        ("🔭","M4","CAPM & Beta",        "4. Riesgo y CAPM"),
        ("🛡️","M5","VaR / CVaR",        "5. Valor en Riesgo (VaR)"),
        ("💎","M6","Optimización",       "6. Optimización"),
        ("🔔","M7","Señales y Alertas",  "7. Señales y Alertas"),
        ("🌍","M8","Macroeconomía",      "8. Macro y Benchmark"),
    ]
    for row_tiles in [tiles[:4], tiles[4:]]:
        cols = st.columns(4)
        for col, (icon, mid, label, target) in zip(cols, row_tiles):
            with col:
                st.markdown(f"""
                <div class='mod-tile'>
                    <div class='mod-icon'>{icon}</div>
                    <div class='mod-id'>{mid}</div>
                    <div class='mod-name'>{label}</div>
                </div>""", unsafe_allow_html=True)
                if st.button(f"Abrir {mid}", key=f"home_{mid}"):
                    set_page(target)

    st.markdown("---")
    st.markdown("""
    <div class='interp-box interp-blue'>
        <b>¿Cómo usar el dashboard?</b><br>
        1. Selecciona el activo que quieres analizar en la barra lateral izquierda.<br>
        2. Elige el período de análisis (mínimo 90 días).<br>
        3. Navega entre los 8 módulos según lo que necesites explorar.<br>
        4. Usa el botón <b>◀</b> de la barra lateral para ocultarla y tener más espacio.
    </div>""", unsafe_allow_html=True)

# ===========================================================================
#  TALLER GAUSS
# ===========================================================================
elif page == "Taller Gauss":
    st.markdown("<div class='rl-h1'>Taller Gauss — Módulos M1 a M4</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>Análisis técnico, estadístico y de riesgo sistemático.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, icon, mid, title, desc, target, key in [
        (c1,"📈","M1","Análisis Técnico","Indicadores: SMA, EMA, RSI, MACD, Bollinger Bands y Estocástico.","1. Análisis Técnico","bg_m1"),
        (c2,"📉","M2","Rendimientos","Estadística descriptiva, pruebas de normalidad y comparativa de retornos.","2. Rendimientos","bg_m2"),
        (c1,"🌊","M3","Volatilidad GARCH","Comparativa de modelos ARCH, GARCH(1,1) y EGARCH(1,1).","3. Volatilidad (GARCH)","bg_m3"),
        (c2,"🔭","M4","CAPM & Beta","Riesgo sistemático, Alpha de Jensen y retorno esperado ajustado al riesgo.","4. Riesgo y CAPM","bg_m4"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div style='font-size:1.7rem'>{icon}</div>
                <h3 style='margin:8px 0 4px;font-size:1rem'>{mid} — {title}</h3>
                <p style='color:{C["muted"]};margin:0 0 12px;font-size:0.85rem'>{desc}</p>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Explorar {mid}", key=key):
                set_page(target)

# ===========================================================================
#  GESTIÓN DE RIESGO
# ===========================================================================
elif page == "Gestion Riesgo":
    st.markdown("<div class='rl-h1'>Gestión de Riesgo — Módulos M5 a M8</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>VaR, optimización de portafolio, señales automáticas y contexto macroeconómico.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, icon, mid, title, desc, target, key in [
        (c1,"🛡️","M5","VaR / CVaR","Valor en Riesgo histórico, paramétrico y Monte Carlo con nivel de confianza configurable.","5. Valor en Riesgo (VaR)","br_m5"),
        (c2,"💎","M6","Optimización","Frontera eficiente de Markowitz con 5,000 portafolios simulados.","6. Optimización","br_m6"),
        (c1,"🔔","M7","Señales","Alertas automáticas de compra/venta basadas en RSI, MACD y Bollinger.","7. Señales y Alertas","br_m7"),
        (c2,"🌍","M8","Macroeconomía","Indicadores globales de referencia y comparativa de riesgo por activo.","8. Macro y Benchmark","br_m8"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div style='font-size:1.7rem'>{icon}</div>
                <h3 style='margin:8px 0 4px;font-size:1rem'>{mid} — {title}</h3>
                <p style='color:{C["muted"]};margin:0 0 12px;font-size:0.85rem'>{desc}</p>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Explorar {mid}", key=key):
                set_page(target)

# ===========================================================================
#  M1 — ANÁLISIS TÉCNICO
# ===========================================================================
elif page == "1. Análisis Técnico":
    st.markdown(f"<div class='rl-h1'>Análisis Técnico — {ticker_selected}</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>Indicadores de precio y momentum para identificar tendencias y señales de corto plazo.</div>",
                unsafe_allow_html=True)

    with st.spinner("Cargando indicadores técnicos..."):
        data = fetch_api(f"/technical/{ticker_selected}", extra_params=dp)

    if data:
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"])
        last = df.iloc[-1]
        prev = df.iloc[-2]

        rsi_val = float(last["RSI"]) if last["RSI"] is not None else 50.0
        rsi_icon = "🔴" if rsi_val > 70 else ("🟢" if rsi_val < 30 else "🟡")

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        price_delta = float(last["Close"]) - float(prev["Close"])
        c1.metric("Precio actual", f"${last['Close']:.2f}", f"{price_delta:+.2f}")
        c2.metric("RSI (14)", f"{rsi_icon} {rsi_val:.1f}", help="<30 sobreventa · 30–70 neutral · >70 sobrecompra")
        c3.metric("Media móvil 20d (SMA)", f"${last['SMA_20']:.2f}" if last["SMA_20"] else "N/A")
        c4.metric("Media móvil exp. 20d (EMA)", f"${last['EMA_20']:.2f}" if last["EMA_20"] else "N/A")

        # Gráfico principal
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.03,
            subplot_titles=("Precio con Indicadores", "RSI — Fuerza Relativa (14 períodos)", "MACD — Momentum"),
        )

        fig.add_trace(go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Precio",
            increasing_line_color=C["green"], decreasing_line_color=C["red"],
            increasing_fillcolor=C["green"], decreasing_fillcolor=C["red"],
        ), row=1, col=1)

        for name, col_key, dash in [("SMA 20", "SMA_20", "dot"), ("EMA 20", "EMA_20", "solid")]:
            clr = C["orange"] if "SMA" in name else C["blue"]
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col_key], name=name,
                                     line=dict(color=clr, width=1.5, dash=dash)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="Bollinger Sup.",
                                 line=dict(color=C["pink"], width=1, dash="dash"), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="Bollinger Inf.",
                                 line=dict(color=C["pink"], width=1, dash="dash"),
                                 fill="tonexty", fillcolor="rgba(221,94,137,0.06)"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI",
                                 line=dict(color=C["purple"], width=2), showlegend=False), row=2, col=1)
        for y_val, color in [(70, C["red"]), (30, C["green"])]:
            fig.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1, row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor=C["red"],   opacity=0.04, line_width=0, row=2, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor=C["green"], opacity=0.04, line_width=0, row=2, col=1)

        colors_hist = [C["green"] if (v or 0) >= 0 else C["red"] for v in df["MACD_Hist"]]
        fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], name="MACD Hist.",
                             marker_color=colors_hist, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Line"], name="MACD",
                                 line=dict(color=C["pink"], width=1.5), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal",
                                 line=dict(color=C["orange"], width=1.5, dash="dot"), showlegend=False), row=3, col=1)

        fig.update_layout(
            **PLOTLY_BASE, height=680,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                        font=dict(color=C["text"], size=11)),
        )
        fig.update_yaxes(row=2, col=1, range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        # Interpretación dinámica
        st.markdown(interp(interp_rsi(rsi_val, ticker_selected)), unsafe_allow_html=True)

        # Estocástico (opcional)
        with st.expander("📊 Oscilador Estocástico — ¿Cuándo está agotado el movimiento?"):
            st.markdown(
                interp("El Estocástico compara el precio de cierre con el rango de precios de los últimos 14 días. "
                       "Valores sobre 80 = sobrecompra; bajo 20 = sobreventa. Es útil para confirmar señales del RSI.", "interp-blue"),
                unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"], name="%K",
                                      line=dict(color=C["pink"], width=2)))
            fig2.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"], name="%D (suavizado)",
                                      line=dict(color=C["orange"], width=1.5, dash="dot")))
            for y_val, color in [(80, C["red"]), (20, C["green"])]:
                fig2.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1)
            fig2.add_hrect(y0=80, y1=100, fillcolor=C["red"],   opacity=0.04, line_width=0)
            fig2.add_hrect(y0=0,  y1=20,  fillcolor=C["green"], opacity=0.04, line_width=0)
            apply_style(fig2, 280)
            fig2.update_yaxes(range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)
    else:
        render_api_error()

# ===========================================================================
#  M2 — RENDIMIENTOS
# ===========================================================================
elif page == "2. Rendimientos":
    st.markdown(f"<div class='rl-h1'>Estadística de Rendimientos — {ticker_selected}</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>¿Cuánto gana o pierde el activo en promedio y qué tan dispersos son esos movimientos?</div>",
                unsafe_allow_html=True)

    with st.spinner("Calculando estadísticas de rendimientos..."):
        res = fetch_api(f"/returns/{ticker_selected}", extra_params=dp)

    if res:
        s   = res["stats"]
        nd  = res["normality"]
        pd_ = res["plot_data"]

        # KPIs principales
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi("Rendimiento Diario Promedio",
                        f"{s['Media']*100:.3f}", "%",
                        desc="Ganancia o pérdida típica en un día de mercado"), unsafe_allow_html=True)
        c2.markdown(kpi("Volatilidad Diaria",
                        f"{s['Desviación Estándar']*100:.2f}", "%",
                        color=C["orange"],
                        desc="Dispersión típica de los rendimientos diarios"), unsafe_allow_html=True)
        c3.markdown(kpi("Asimetría",
                        f"{s['Asimetría (Skewness)']:.3f}", "",
                        color=C["blue"],
                        desc="0 = simétrico · negativo = cola izquierda (pérdidas extremas)"), unsafe_allow_html=True)
        c4.markdown(kpi("Curtosis",
                        f"{s['Curtosis']:.3f}", "",
                        color=C["purple"],
                        desc=">3 = colas pesadas (más eventos extremos que la distribución normal"), unsafe_allow_html=True)

        col_chart, col_stats = st.columns([3, 2])

        with col_chart:
            returns_arr = pd_.get("Simple_Returns", [])
            mean_r = float(np.mean(returns_arr))
            std_r  = float(np.std(returns_arr))
            x_norm = np.linspace(min(returns_arr), max(returns_arr), 200)
            from scipy.stats import norm
            counts, bins = np.histogram(returns_arr, bins=60)
            bw = bins[1] - bins[0]
            y_norm = norm.pdf(x_norm, mean_r, std_r) * len(returns_arr) * bw

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=returns_arr, nbinsx=60, name="Rendimientos observados",
                                       marker_color=C["pink"], opacity=0.72))
            fig.add_trace(go.Scatter(x=x_norm.tolist(), y=y_norm.tolist(),
                                     name="Distribución normal teórica",
                                     line=dict(color=C["sidebar"], width=2.5, dash="dot")))
            fig.update_layout(**PLOTLY_BASE, height=340,
                              title=f"Distribución de Rendimientos Diarios — {ticker_selected}",
                              bargap=0.02, legend=_LEGEND)
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            # Pruebas de normalidad (siempre visibles)
            jb  = nd["Jarque-Bera"]
            sw  = nd["Shapiro-Wilk"]
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Prueba Jarque-Bera</div>
                <div style='margin:8px 0 4px'><b>Estadístico:</b> {jb['stat']:.4f}</div>
                <div style='margin-bottom:8px'><b>p-valor:</b> {jb['p_value']:.4f}</div>
                {norm_badge(jb['p_value'])}
                <div class='kpi-desc' style='margin-top:10px'>
                    Compara la asimetría y curtosis con la normal.<br>
                    p {'> 0.05 → compatible con normalidad.' if jb['p_value']>0.05 else '≤ 0.05 → se rechaza normalidad.'}
                </div>
            </div>
            <div class='kpi-card' style='margin-top:0'>
                <div class='kpi-label'>Prueba Shapiro-Wilk</div>
                <div style='margin:8px 0 4px'><b>Estadístico:</b> {sw['stat']:.4f}</div>
                <div style='margin-bottom:8px'><b>p-valor:</b> {sw['p_value']:.4f}</div>
                {norm_badge(sw['p_value'])}
            </div>""", unsafe_allow_html=True)

        # Interpretación
        st.markdown(interp(interp_returns(s["Asimetría (Skewness)"], s["Curtosis"], ticker_selected)),
                    unsafe_allow_html=True)

        # Estadísticas detalladas (colapsable)
        with st.expander("📋 Ver tabla de estadísticas descriptivas completa"):
            rows = [
                ("Mínimo diario",           f"{s['Mínimo']*100:.3f}%"),
                ("Máximo diario",           f"{s['Máximo']*100:.3f}%"),
                ("Media diaria",            f"{s['Media']*100:.4f}%"),
                ("Desv. estándar",          f"{s['Desviación Estándar']*100:.4f}%"),
                ("Volatilidad anualizada",  f"{s['Desviación Estándar']*100*(252**0.5):.2f}%"),
                ("Asimetría (Skewness)",    f"{s['Asimetría (Skewness)']:.4f}"),
                ("Curtosis (exceso)",       f"{s['Curtosis']:.4f}"),
                ("Observaciones",           str(int(s["Conteo"]))),
            ]
            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            for lbl, val in rows:
                st.markdown(
                    f"<div class='stat-row'><span class='stat-label'>{lbl}</span>"
                    f"<span class='stat-value'>{val}</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📐 Gráfico Q-Q — ¿Se distribuyen normalmente los rendimientos?"):
            st.markdown(
                interp("En este gráfico, si los puntos siguen la línea diagonal, los rendimientos son normales. "
                       "Puntos alejados en los extremos indican colas pesadas (fat tails) — "
                       "mayor probabilidad de movimientos extremos de lo que la distribución normal predice.", "interp-blue"),
                unsafe_allow_html=True)
            returns_arr = np.array(pd_["Simple_Returns"])
            returns_arr = returns_arr[np.isfinite(returns_arr)]
            if len(returns_arr) > 10:
                n = len(returns_arr)
                probs = np.linspace(0.01, 0.99, n)
                th_q  = scipy_stats.norm.ppf(probs)
                em_q  = np.interp(probs, np.linspace(0, 1, n), np.sort(returns_arr))
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(x=th_q.tolist(), y=em_q.tolist(),
                                            mode="markers", name="Datos empíricos",
                                            marker=dict(color=C["pink"], size=3, opacity=0.55)))
                lim = [float(th_q.min()), float(th_q.max())]
                fig_qq.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                            name="Referencia normal",
                                            line=dict(color=C["sidebar"], dash="dash", width=2)))
                apply_style(fig_qq, 340)
                fig_qq.update_layout(
                    xaxis_title="Cuantiles teóricos (distribución normal)",
                    yaxis_title="Cuantiles observados (retornos reales)",
                )
                st.plotly_chart(fig_qq, use_container_width=True)

        with st.expander("📈 Comparativa: Rendimientos Simples vs Logarítmicos"):
            st.markdown(
                interp("Los rendimientos simples muestran la ganancia/pérdida porcentual directa. "
                       "Los logarítmicos son más convenientes para análisis estadístico y suelen ser casi idénticos para movimientos pequeños.", "interp-blue"),
                unsafe_allow_html=True)
            dates = pd_.get("Dates", [])
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=("Rendimientos Simples (%)", "Rendimientos Logarítmicos (%)"),
                                 vertical_spacing=0.08)
            fig2.add_trace(go.Scatter(x=dates, y=pd_["Simple_Returns"],
                                      line=dict(color=C["pink"], width=1), name="Simple"), row=1, col=1)
            fig2.add_trace(go.Scatter(x=dates, y=pd_["Log_Returns"],
                                      line=dict(color=C["orange"], width=1), name="Log"), row=2, col=1)
            for r in [1, 2]:
                fig2.update_yaxes(tickformat=".2%", row=r, col=1)
            apply_style(fig2, 380)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        render_api_error()

# ===========================================================================
#  M3 — VOLATILIDAD GARCH
# ===========================================================================
elif page == "3. Volatilidad (GARCH)":
    st.markdown(f"<div class='rl-h1'>Modelos de Volatilidad — {ticker_selected}</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>Comparamos tres modelos que miden cómo cambia el riesgo del activo a lo largo del tiempo.</div>",
                unsafe_allow_html=True)

    with st.spinner("Ajustando modelos ARCH/GARCH (puede tardar unos segundos)..."):
        res = fetch_api(f"/volatility/{ticker_selected}", timeout=60, extra_params=dp)

    if res:
        best_aic = min(res.items(), key=lambda x: x[1]["AIC"])[0]

        st.markdown("<div class='section-header'>Comparativa de modelos</div>", unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        for col, (name, vals) in zip([mc1, mc2, mc3], res.items()):
            is_best = name == best_aic
            border  = f"border-top:3px solid {C['pink']};" if is_best else f"border-top:3px solid {C['border']};"
            badge   = ("<span style='background:#fce7f3;color:#9d174d;padding:2px 8px;"
                       "border-radius:10px;font-size:0.72rem;font-weight:700'>✓ MEJOR AIC</span> "
                       if is_best else "")
            col.markdown(f"""
            <div class='kpi-card' style='{border}'>
                <div class='kpi-label'>{name} {badge}</div>
                <div class='stat-row'><span class='stat-label'>AIC</span>
                    <span class='stat-value'>{vals['AIC']:.2f}</span></div>
                <div class='stat-row'><span class='stat-label'>BIC</span>
                    <span class='stat-value'>{vals['BIC']:.2f}</span></div>
                <div class='stat-row'><span class='stat-label'>Log-Verosimilitud</span>
                    <span class='stat-value'>{vals['LogL']:.2f}</span></div>
            </div>""", unsafe_allow_html=True)

        st.markdown(
            interp("El modelo con <b>menor AIC/BIC</b> tiene mejor balance entre precisión y complejidad. "
                   "GARCH(1,1) es el más usado en finanzas; EGARCH captura además el efecto asimétrico "
                   "(las caídas generan más volatilidad que las subidas de igual magnitud).", "interp-blue"),
            unsafe_allow_html=True)

        vol_series = res.get("GARCH(1,1)", {}).get("Volatility", [])
        if vol_series:
            vol_mean = float(np.mean(vol_series))
            st.markdown("<div class='section-header'>Volatilidad condicional — GARCH(1,1)</div>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=vol_series, name="Volatilidad condicional",
                line=dict(color=C["pink"], width=2),
                fill="tozeroy", fillcolor="rgba(221,94,137,0.1)",
            ))
            fig.add_hline(y=vol_mean, line_dash="dot", line_color=C["blue"],
                          annotation_text=f"Promedio: {vol_mean:.2%}",
                          annotation_position="right",
                          annotation_font=dict(color=C["text"], size=11))
            fig.update_layout(**PLOTLY_BASE, height=360,
                              xaxis_title="Días de negociación",
                              yaxis_title="Volatilidad diaria condicional",
                              yaxis_tickformat=".2%",
                              legend=_LEGEND)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(interp(interp_volatility(vol_mean, ticker_selected, best_aic)), unsafe_allow_html=True)
    else:
        render_api_error()

# ===========================================================================
#  M4 — CAPM & BETA
# ===========================================================================
elif page == "4. Riesgo y CAPM":
    st.markdown(f"<div class='rl-h1'>CAPM & Riesgo Sistemático — {ticker_selected}</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>¿Cuánto riesgo de mercado tiene este activo y qué retorno debería compensarlo?</div>",
                unsafe_allow_html=True)

    with st.spinner("Calculando CAPM..."):
        res = fetch_api(f"/risk/{ticker_selected}", extra_params=dp)

    if res:
        capm  = res["capm"]
        beta  = capm["Beta"]
        alpha = capm["Alpha"]
        r2    = capm["R_Squared"]
        exp_a = capm["Expected_Return_Annual"]
        cls   = capm["Classification"]

        cls_color = C["red"] if cls == "Agresivo" else (C["green"] if cls == "Defensivo" else C["yellow"])
        cls_icon  = "🔴" if cls == "Agresivo" else ("🟢" if cls == "Defensivo" else "🟡")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beta (β)", f"{beta:.4f}", help="< 0.8 Defensivo · 0.8–1.2 Neutro · > 1.2 Agresivo")
        c2.metric("Alpha (α)", f"{alpha:.5f}", help="Retorno por encima o debajo de lo que predice el CAPM")
        c3.metric("R² (ajuste del modelo)", f"{r2:.3f}", help="Qué % de la varianza explica el S&P 500")
        c4.metric("Retorno esperado anual", f"{exp_a*100:.2f}%")

        col_info, col_gauge = st.columns([1, 1])
        with col_info:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Clasificación del activo</div>
                <div style='font-size:2.2rem;font-weight:700;color:{cls_color};margin:10px 0'>
                    {cls_icon} {cls}
                </div>
                <div class='stat-row'><span class='stat-label'>Beta (β)</span>
                    <span class='stat-value'>{beta:.4f}</span></div>
                <div class='stat-row'><span class='stat-label'>Alpha diario</span>
                    <span class='stat-value'>{alpha:.6f}</span></div>
                <div class='stat-row'><span class='stat-label'>R² del modelo</span>
                    <span class='stat-value'>{r2*100:.2f}%</span></div>
                <div class='stat-row'><span class='stat-label'>Retorno esperado anual</span>
                    <span class='stat-value'>{exp_a*100:.2f}%</span></div>
            </div>""", unsafe_allow_html=True)

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(beta, 3),
                delta={"reference": 1.0, "valueformat": ".3f"},
                gauge={
                    "axis": {"range": [0, 2.5], "tickwidth": 1,
                             "tickfont": dict(color=C["text"], size=11)},
                    "bar":  {"color": cls_color, "thickness": 0.28},
                    "steps": [
                        {"range": [0, 0.8],   "color": "rgba(16,185,129,0.12)"},
                        {"range": [0.8, 1.2], "color": "rgba(245,158,11,0.10)"},
                        {"range": [1.2, 2.5], "color": "rgba(239,68,68,0.12)"},
                    ],
                    "threshold": {"line": {"color": C["sidebar"], "width": 3},
                                  "thickness": 0.8, "value": 1.0},
                },
                title={"text": "Beta vs. Mercado (β=1 = S&P 500)", "font": {"size": 13, "color": C["text"]}},
                number={"font": {"size": 38, "color": cls_color}},
            ))
            fig.update_layout(**PLOTLY_BASE, height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(interp(interp_capm(beta, alpha, r2, cls, ticker_selected)), unsafe_allow_html=True)

        with st.expander("📊 Dispersión CAPM — Retorno del activo vs. retorno del mercado"):
            st.markdown(
                interp("Cada punto es un día de mercado. La línea de regresión muestra la relación entre el movimiento "
                       "del S&P 500 y el de este activo. La pendiente de esa línea es la Beta.", "interp-blue"),
                unsafe_allow_html=True)
            with st.spinner("Cargando dispersión..."):
                ret_raw   = fetch_api(f"/returns/{ticker_selected}", extra_params=dp)
                bench_raw = fetch_api("/returns/%5EGSPC", extra_params=dp)
            if ret_raw and bench_raw:
                sr = pd.Series(ret_raw["plot_data"]["Simple_Returns"],
                               index=pd.to_datetime(ret_raw["plot_data"]["Dates"]))
                br = pd.Series(bench_raw["plot_data"]["Simple_Returns"],
                               index=pd.to_datetime(bench_raw["plot_data"]["Dates"]))
                common = sr.index.intersection(br.index)
                if len(common) >= 30:
                    src, brc = sr.loc[common].values, br.loc[common].values
                    slope, intercept, r_val, _, _ = scipy_stats.linregress(brc, src)
                    xlim = [float(brc.min()), float(brc.max())]
                    ylim = [intercept + slope * x for x in xlim]
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(x=brc.tolist(), y=src.tolist(),
                                               mode="markers", name="Días de mercado",
                                               marker=dict(color=C["pink"], size=4, opacity=0.32)))
                    fig_s.add_trace(go.Scatter(x=xlim, y=ylim, mode="lines",
                                               name=f"Regresión OLS  β={slope:.3f}  α={intercept:.5f}",
                                               line=dict(color=C["sidebar"], width=2.5)))
                    apply_style(fig_s, 380)
                    fig_s.update_layout(
                        xaxis_title="Retorno diario — S&P 500 (mercado)",
                        yaxis_title=f"Retorno diario — {ticker_selected}",
                        xaxis_tickformat=".2%", yaxis_tickformat=".2%",
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
    else:
        render_api_error()

# ===========================================================================
#  M5 — VaR / CVaR
# ===========================================================================
elif page == "5. Valor en Riesgo (VaR)":
    st.markdown(f"<div class='rl-h1'>Valor en Riesgo (VaR & CVaR) — {ticker_selected}</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>¿Cuánto puede perder este activo en un día adverso?</div>", unsafe_allow_html=True)

    # Controles VaR con validación
    st.markdown("<div class='section-header'>Parámetros de Simulación</div>", unsafe_allow_html=True)
    vp1, vp2, vp3 = st.columns([2, 2, 1])
    with vp1:
        var_conf = st.slider(
            "Nivel de confianza",
            min_value=0.80, max_value=0.99, step=0.01,
            value=st.session_state.var_confidence, format="%.2f",
            help="Probabilidad de que la pérdida real no supere el VaR calculado.",
        )
    with vp2:
        var_iter = st.select_slider(
            "Simulaciones Monte Carlo",
            options=[1_000, 2_000, 5_000, 10_000, 20_000, 50_000],
            value=st.session_state.var_iterations,
            help="Más simulaciones = resultado más estable, pero más lento.",
        )
    with vp3:
        recalc = st.button("🔄 Recalcular", use_container_width=True)

    st.session_state.var_confidence = var_conf
    st.session_state.var_iterations = var_iter

    var_params = f"confidence={var_conf}&n_simulations={var_iter}"
    full_params = f"{dp}&{var_params}" if dp else var_params

    with st.spinner("Calculando métricas de riesgo..."):
        res = fetch_api(f"/risk/{ticker_selected}", timeout=40, extra_params=full_params)

    if res:
        var = res["var"]
        conf_pct = int(var_conf * 100)

        methods = [
            ("Histórico",   var["Historico"],   C["pink"],   "📊",
             "Basado en datos reales del activo. El método más directo."),
            ("Paramétrico", var["Parametrico"],  C["blue"],   "📐",
             "Asume distribución normal. Más rápido, pero subestima colas pesadas."),
            ("Monte Carlo", var["Montecarlo"],   C["orange"], "🎲",
             f"Simula {var_iter:,} escenarios aleatorios. Más flexible."),
        ]

        st.markdown("<div class='section-header'>Resultados por metodología</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, (name, vals, color, icon, note) in zip([c1, c2, c3], methods):
            col.markdown(f"""
            <div class='kpi-card' style='border-top:3px solid {color}'>
                <div class='kpi-label'>{icon} {name}</div>
                <div style='margin:10px 0 4px'>
                    <span class='kpi-label'>VaR {conf_pct}%</span><br>
                    <span class='kpi-value' style='color:{color}'>{vals['VaR']*100:.2f}
                        <span class='kpi-unit'>%</span></span>
                </div>
                <div style='margin-top:10px'>
                    <span class='kpi-label'>CVaR {conf_pct}%</span><br>
                    <span style='font-size:1.3rem;font-weight:700;color:{color};opacity:0.8'>
                        {vals['CVaR']*100:.2f}<span class='kpi-unit'>%</span></span>
                </div>
                <div class='kpi-desc' style='margin-top:10px'>{note}</div>
            </div>""", unsafe_allow_html=True)

        # Interpretación dinámica
        st.markdown(
            interp(interp_var(var["Historico"]["VaR"], var["Historico"]["CVaR"], var_conf, ticker_selected)),
            unsafe_allow_html=True)

        # Gráfico comparativo
        st.markdown("<div class='section-header'>Comparativa visual</div>", unsafe_allow_html=True)
        labels    = ["Histórico", "Paramétrico", "Monte Carlo"]
        var_vals  = [var[k]["VaR"]*100  for k in ["Historico","Parametrico","Montecarlo"]]
        cvar_vals = [var[k]["CVaR"]*100 for k in ["Historico","Parametrico","Montecarlo"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(name=f"VaR {conf_pct}%",  x=labels, y=var_vals,
                             marker_color=C["pink"],
                             text=[f"{v:.2f}%" for v in var_vals], textposition="outside"))
        fig.add_trace(go.Bar(name=f"CVaR {conf_pct}%", x=labels, y=cvar_vals,
                             marker_color=C["sidebar"], opacity=0.75,
                             text=[f"{v:.2f}%" for v in cvar_vals], textposition="outside"))
        fig.update_layout(**PLOTLY_BASE, barmode="group", height=360,
                          yaxis_title="Pérdida estimada diaria (%)",
                          yaxis_tickformat=".2f", yaxis_ticksuffix="%",
                          legend=_LEGEND)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            interp(f"<b>VaR</b> = pérdida máxima que no se supera el {conf_pct}% de los días. "
                   f"<b>CVaR</b> = pérdida promedio en el {100-conf_pct}% de días más adversos. "
                   "Todos los valores son positivos (representan pérdidas). "
                   "A mayor confianza, mayor es el VaR estimado.", "interp-blue"),
            unsafe_allow_html=True)
    else:
        render_api_error()

# ===========================================================================
#  M6 — OPTIMIZACIÓN
# ===========================================================================
elif page == "6. Optimización":
    st.markdown("<div class='rl-h1'>Optimización de Portafolio — Markowitz</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>¿Cómo distribuir el capital entre los activos para maximizar el retorno por unidad de riesgo?</div>",
                unsafe_allow_html=True)

    with st.spinner("Optimizando portafolio (simulando 5,000 escenarios)..."):
        res = fetch_api("/portfolio/optimize", timeout=90, extra_params=dp)

    if res:
        ms = res["Max_Sharpe"]
        mv = res["Min_Volatility"]

        c1, c2 = st.columns(2)
        c1.markdown(kpi("Retorno esperado — Máx. Sharpe",    f"{ms['Return']*100:.2f}", "%",
                        desc="Portafolio con mejor relación retorno/riesgo"), unsafe_allow_html=True)
        c1.markdown(kpi("Volatilidad — Máx. Sharpe",         f"{ms['Volatility']*100:.2f}", "%",
                        color=C["orange"]), unsafe_allow_html=True)
        c2.markdown(kpi("Retorno esperado — Mín. Volatilidad", f"{mv['Return']*100:.2f}", "%",
                        color=C["blue"],
                        desc="Portafolio con el menor riesgo posible"), unsafe_allow_html=True)
        c2.markdown(kpi("Volatilidad — Mín. Volatilidad",    f"{mv['Volatility']*100:.2f}", "%",
                        color=C["green"]), unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Distribución óptima", "⚖️ Comparativa de pesos", "🔗 Correlaciones"])

        with tab1:
            st.markdown(
                interp("Cada gráfico muestra cómo distribuir el capital. "
                       "El portafolio de <b>Máx. Sharpe</b> busca el mejor retorno ajustado al riesgo. "
                       "El de <b>Mín. Volatilidad</b> prioriza la estabilidad sobre el retorno.", "interp-blue"),
                unsafe_allow_html=True)
            tickers_list  = list(ms["Weights"].keys())
            colors_pie    = [asset_color(t) for t in tickers_list]

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Máximo Sharpe", "Mínima Volatilidad"),
                                specs=[[{"type":"pie"},{"type":"pie"}]])
            for col_idx, weights in enumerate([ms["Weights"], mv["Weights"]], 1):
                fig.add_trace(go.Pie(
                    labels=list(weights.keys()),
                    values=list(weights.values()),
                    hole=0.44, name="",
                    marker_colors=colors_pie,
                    textfont=dict(size=12, color=C["text"]),
                    hovertemplate="<b>%{label}</b><br>Peso: %{percent}<extra></extra>",
                ), row=1, col=col_idx)
            apply_style(fig, 360)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Multiselect para controlar qué activos comparar
            sel_tickers = st.multiselect(
                "Selecciona activos para comparar",
                options=tickers_list,
                default=tickers_list,
                help="Puedes agregar o quitar activos de la comparación.",
            )
            if sel_tickers:
                w_ms_sel  = {t: ms["Weights"][t]*100  for t in sel_tickers}
                w_mv_sel  = {t: mv["Weights"][t]*100  for t in sel_tickers}
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=sel_tickers, y=[w_ms_sel[t] for t in sel_tickers],
                    name="Máx. Sharpe",
                    marker_color=[asset_color(t) for t in sel_tickers],
                    text=[f"{v:.1f}%" for v in w_ms_sel.values()],
                    textposition="outside",
                ))
                fig_bar.add_trace(go.Bar(
                    x=sel_tickers, y=[w_mv_sel[t] for t in sel_tickers],
                    name="Mín. Volatilidad",
                    marker_color=[asset_color(t) for t in sel_tickers],
                    marker_pattern_shape="x",
                    opacity=0.7,
                    text=[f"{v:.1f}%" for v in w_mv_sel.values()],
                    textposition="outside",
                ))
                fig_bar.update_layout(**PLOTLY_BASE, barmode="group", height=360,
                                      yaxis_title="Peso asignado (%)", legend=_LEGEND)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Selecciona al menos un activo para mostrar la comparativa.")

        with tab3:
            st.markdown(
                interp("La correlación mide cómo se mueven dos activos juntos. "
                       "<b>+1</b> = se mueven igual · <b>0</b> = sin relación · <b>-1</b> = se mueven opuesto. "
                       "Para diversificar bien, buscamos activos con correlaciones bajas o negativas.", "interp-blue"),
                unsafe_allow_html=True)
            corr = pd.DataFrame(res["Correlation"])
            fig_corr = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale=[[0, C["blue"]], [0.5, "#ffffff"], [1, C["pink"]]],
                zmin=-1, zmax=1, aspect="auto",
            )
            fig_corr.update_traces(textfont=dict(size=13, color=C["text"]))
            fig_corr.update_layout(**PLOTLY_BASE, height=360)
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        render_api_error()

# ===========================================================================
#  M7 — SEÑALES Y ALERTAS
# ===========================================================================
elif page == "7. Señales y Alertas":
    st.markdown(f"<div class='rl-h1'>Señales y Alertas — {ticker_selected}</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>Alertas automáticas basadas en indicadores técnicos, con explicación de por qué se activan.</div>",
                unsafe_allow_html=True)

    with st.spinner("Generando señales..."):
        res  = fetch_api(f"/signals/{ticker_selected}", extra_params=dp)
        tech = fetch_api(f"/technical/{ticker_selected}", extra_params=dp)

    if res:
        signals = res.get("signals", [])

        if not signals:
            st.markdown("""
            <div class='sig-none'>
                <div style='font-size:2rem'>✅</div>
                <h3 style='margin:8px 0 4px'>Sin alertas activas</h3>
                <p style='margin:0'>Los indicadores no muestran señales extremas de sobrecompra, sobreventa ni cruces en este momento.</p>
            </div>""", unsafe_allow_html=True)
        else:
            buy_sigs  = [s for s in signals if s["type"] == "Buy"]
            sell_sigs = [s for s in signals if s["type"] == "Sell"]

            if buy_sigs:
                st.markdown("<div class='section-header'>Señales de Compra</div>", unsafe_allow_html=True)
                for sig in buy_sigs:
                    expl = sig.get("explanation", "")
                    st.markdown(f"""
                    <div class='sig-buy'>
                        ✅ <strong>{sig['id']}</strong> — {sig['msg']}
                        {"<div class='sig-explanation'>"+expl+"</div>" if expl else ""}
                    </div>""", unsafe_allow_html=True)

            if sell_sigs:
                st.markdown("<div class='section-header'>Señales de Venta / Precaución</div>", unsafe_allow_html=True)
                for sig in sell_sigs:
                    expl = sig.get("explanation", "")
                    st.markdown(f"""
                    <div class='sig-sell'>
                        ⚠️ <strong>{sig['id']}</strong> — {sig['msg']}
                        {"<div class='sig-explanation'>"+expl+"</div>" if expl else ""}
                    </div>""", unsafe_allow_html=True)

        # Gauges de indicadores
        if tech:
            last    = tech[-1]
            rsi     = float(last.get("RSI") or 50)
            macd_h  = float(last.get("MACD_Hist") or 0)
            close   = float(last.get("Close") or 0)
            bb_up   = float(last.get("BB_Upper") or 0)
            bb_low  = float(last.get("BB_Lower") or 0)
            bb_pos  = (close - bb_low) / (bb_up - bb_low) * 100 if (bb_up - bb_low) != 0 else 50

            rsi_status  = "Sobrecompra" if rsi > 70 else ("Sobreventa" if rsi < 30 else "Neutral")
            macd_status = "Alcista ▲"   if macd_h > 0 else "Bajista ▼"
            bb_status   = "Banda sup." if bb_pos > 80 else ("Banda inf." if bb_pos < 20 else "Zona central")

            st.markdown("<div class='section-header'>Indicadores en tiempo real</div>", unsafe_allow_html=True)
            gc1, gc2, gc3 = st.columns(3)

            rsi_c = C["red"] if rsi > 70 else (C["green"] if rsi < 30 else C["yellow"])
            fig_rsi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(rsi, 1),
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                             "tickfont": dict(color=C["text"], size=10)},
                    "bar": {"color": rsi_c, "thickness": 0.28},
                    "steps": [
                        {"range": [0, 30],  "color": "rgba(16,185,129,0.12)"},
                        {"range": [30, 70], "color": "rgba(245,158,11,0.07)"},
                        {"range": [70,100], "color": "rgba(239,68,68,0.12)"},
                    ],
                    "threshold": {"line":{"color":rsi_c,"width":3},"thickness":0.8,"value":rsi},
                },
                title={"text": f"RSI (14) — {rsi_status}", "font": {"size": 13, "color": C["text"]}},
                number={"font": {"size": 30, "color": rsi_c}},
            ))
            fig_rsi.update_layout(**PLOTLY_BASE, height=230, margin=dict(t=55,b=10,l=20,r=20))
            gc1.plotly_chart(fig_rsi, use_container_width=True)
            gc1.caption("< 30 = sobreventa · 30–70 = neutral · > 70 = sobrecompra")

            macd_c = C["green"] if macd_h > 0 else C["red"]
            fig_m = go.Figure(go.Indicator(
                mode="number+delta",
                value=round(macd_h, 5),
                delta={"reference": 0, "valueformat": ".5f"},
                title={"text": f"MACD Histograma — {macd_status}", "font": {"size": 13, "color": C["text"]}},
                number={"font": {"size": 30, "color": macd_c}, "valueformat": ".5f"},
            ))
            fig_m.update_layout(**PLOTLY_BASE, height=230, margin=dict(t=55,b=10,l=20,r=20))
            gc2.plotly_chart(fig_m, use_container_width=True)
            gc2.caption("Positivo = momentum alcista · Negativo = momentum bajista")

            bb_c = C["red"] if bb_pos > 80 else (C["green"] if bb_pos < 20 else C["yellow"])
            fig_bb = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(bb_pos, 1),
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                             "tickfont": dict(color=C["text"], size=10)},
                    "bar": {"color": bb_c, "thickness": 0.28},
                    "steps": [
                        {"range": [0, 20],   "color": "rgba(16,185,129,0.12)"},
                        {"range": [20, 80],  "color": "rgba(245,158,11,0.07)"},
                        {"range": [80, 100], "color": "rgba(239,68,68,0.12)"},
                    ],
                    "threshold": {"line":{"color":bb_c,"width":3},"thickness":0.8,"value":bb_pos},
                },
                title={"text": f"Posición Bollinger (%B) — {bb_status}", "font": {"size": 13, "color": C["text"]}},
                number={"font": {"size": 30, "color": bb_c}, "suffix": "%"},
            ))
            fig_bb.update_layout(**PLOTLY_BASE, height=230, margin=dict(t=55,b=10,l=20,r=20))
            gc3.plotly_chart(fig_bb, use_container_width=True)
            gc3.caption("< 20 = cerca banda inf. · 20–80 = rango normal · > 80 = cerca banda sup.")
    else:
        render_api_error()

# ===========================================================================
#  M8 — MACROECONOMÍA
# ===========================================================================
elif page == "8. Macro y Benchmark":
    st.markdown("<div class='rl-h1'>Macroeconomía & Benchmark</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>Indicadores de referencia para contextualizar el riesgo del portafolio en el entorno global.</div>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, value, desc, color in [
        (c1, "💵", "Tasa Libre de Riesgo", "~4.50%", "T-Bill 13 sem. (^IRX) — usada en Sharpe y CAPM", C["blue"]),
        (c2, "📈", "Benchmark",            "S&P 500", "Índice de referencia del portafolio (^GSPC)",    C["green"]),
        (c3, "🏦", "Inflación (CPI EE.UU.)","~3.2%", "Tasa interanual de referencia",                  C["orange"]),
        (c4, "🌡️", "Índice VIX",           "~15–20", "< 20 = calma · > 30 = estrés de mercado",       C["pink"]),
    ]:
        col.markdown(f"""
        <div class='kpi-card' style='border-top:3px solid {color};text-align:center'>
            <div style='font-size:1.9rem'>{icon}</div>
            <div class='kpi-label' style='margin-top:6px'>{label}</div>
            <div class='kpi-value' style='color:{color};font-size:1.5rem'>{value}</div>
            <div class='kpi-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        interp("Los indicadores de arriba son <b>valores de referencia</b>. "
               "La tasa libre de riesgo (^IRX) se descarga automáticamente del mercado y se usa "
               "en los cálculos de CAPM y Sharpe Ratio del portafolio.", "interp-blue"),
        unsafe_allow_html=True)

    # Tablas de referencia (colapsables)
    with st.expander("📖 Guía de interpretación — Beta y métricas de riesgo"):
        c_l, c_r = st.columns(2)
        with c_l:
            st.markdown("**Clasificación por Beta (β)**")
            st.markdown("""
| Beta (β) | Clasificación | Qué significa |
|:---:|:---:|:---|
| β < 0 | ⬛ Inverso | Se mueve contrario al mercado |
| 0 – 0.8 | 🟢 Defensivo | Amplifica menos las variaciones del mercado |
| 0.8 – 1.2 | 🟡 Neutro | Sigue de cerca al mercado |
| β > 1.2 | 🔴 Agresivo | Amplifica los movimientos — más riesgo y retorno potencial |
""")
        with c_r:
            st.markdown("**Métricas clave**")
            st.markdown("""
| Métrica | Interpretación |
|:---|:---|
| Sharpe Ratio | Mayor = mejor retorno por unidad de riesgo |
| Alpha (Jensen) | > 0: supera al benchmark ajustado por riesgo |
| R² | Qué % de varianza explica el mercado |
| CVaR / ES | Pérdida promedio en los peores escenarios |
""")

    # Comparativa CAPM por activo
    st.markdown("<div class='section-header'>Riesgo sistemático por activo</div>", unsafe_allow_html=True)

    # Multiselect para activos
    sel_tickers_m8 = st.multiselect(
        "Selecciona activos para comparar",
        options=TICKERS,
        default=TICKERS,
        key="m8_multiselect",
        help="Agrega o quita activos del gráfico de comparación.",
    )

    with st.spinner("Cargando métricas CAPM..."):
        risk_rows = []
        for t in sel_tickers_m8:
            rd = fetch_api(f"/risk/{t}", extra_params=dp)
            if rd and "capm" in rd:
                q = rd["capm"]
                risk_rows.append({
                    "Ticker": t,
                    "Beta (β)": round(q["Beta"], 3),
                    "Alpha (α)": round(q["Alpha"], 6),
                    "R²": round(q["R_Squared"], 3),
                    "Retorno Esp. Anual (%)": round(q["Expected_Return_Annual"] * 100, 2),
                    "Clasificación": q["Classification"],
                    "_color": C["red"] if q["Classification"]=="Agresivo"
                              else C["green"] if q["Classification"]=="Defensivo"
                              else C["yellow"],
                })

    if risk_rows:
        # Gráfico primero (más visual)
        fig_beta = go.Figure()
        for row in risk_rows:
            fig_beta.add_trace(go.Bar(
                x=[row["Ticker"]], y=[row["Beta (β)"]],
                name=row["Ticker"],
                marker_color=asset_color(row["Ticker"]),
                text=[f"β={row['Beta (β)']:.3f}"],
                textposition="outside",
                textfont=dict(size=12, color=C["text"]),
                showlegend=True,
            ))
        fig_beta.add_hline(y=1.0, line_dash="dot", line_color="#475569",
                           line_width=1.5,
                           annotation_text="β = 1.0 (mercado)", annotation_position="right",
                           annotation_font=dict(color=C["text"], size=11))
        fig_beta.add_hline(y=1.2, line_dash="dash", line_color=C["red"],   line_width=1)
        fig_beta.add_hline(y=0.8, line_dash="dash", line_color=C["green"], line_width=1)
        fig_beta.update_layout(**PLOTLY_BASE, height=360,
                               xaxis_title="Activo", yaxis_title="Beta (β)",
                               barmode="group", legend=_LEGEND,
                               title="Beta por activo — riesgo sistemático vs. S&P 500")
        st.plotly_chart(fig_beta, use_container_width=True)

        # Tabla detallada (colapsable)
        with st.expander("📋 Ver tabla detallada de métricas CAPM"):
            df_risk = pd.DataFrame(
                [{k: v for k, v in r.items() if k != "_color"} for r in risk_rows]
            ).set_index("Ticker")
            st.dataframe(
                df_risk.style
                .background_gradient(subset=["Beta (β)"],            cmap="RdYlGn_r")
                .background_gradient(subset=["R²"],                   cmap="Blues")
                .format({"Beta (β)": "{:.3f}", "Alpha (α)": "{:.6f}",
                         "R²": "{:.3f}", "Retorno Esp. Anual (%)": "{:.2f}%"}),
                use_container_width=True,
            )

# ===========================================================================
#  M9 — OPORTUNIDADES DE MEJORA
# ===========================================================================
elif page == "9. Oportunidades de Mejora":
    st.markdown("<div class='rl-h1'>Oportunidades de Mejora</div>", unsafe_allow_html=True)
    st.markdown("<div class='rl-sub'>Extensiones naturales de este proyecto que elevarían su capacidad analítica y su valor en producción.</div>",
                unsafe_allow_html=True)

    st.markdown(
        interp("Esta sección presenta <b>propuestas concretas de trabajo futuro</b>. "
               "Cada mejora tiene sentido técnico dentro de la arquitectura actual y ampliaría "
               "significativamente las capacidades de RiskLab USTA.", "interp-blue"),
        unsafe_allow_html=True)

    improvements = [
        ("🤖", "Modelos Predictivos con ML",
         "Aprendizaje automático",
         "Incorporar modelos LSTM o GRU para predicción de series temporales de precios y volatilidad. "
         "También modelos de clasificación para anticipar señales compradoras/vendedoras con mayor precisión."),
        ("📰", "Análisis de Sentimiento",
         "NLP / Datos alternativos",
         "Integrar APIs de noticias financieras (NewsAPI, Bloomberg) y redes sociales (Reddit, Twitter/X) "
         "para construir indicadores de sentimiento que complementen el análisis técnico."),
        ("📊", "Backtesting de Estrategias",
         "Validación histórica",
         "Simular cómo habrían funcionado las señales actuales en el pasado. "
         "Evaluar estrategias con métricas como Sharpe, Calmar Ratio, máximo drawdown y win rate."),
        ("🎯", "Modelos de Factores (Fama-French)",
         "Finanzas cuantitativas avanzadas",
         "Extender el CAPM de un factor (mercado) a tres o cinco factores: tamaño, valor, rentabilidad "
         "y momentum. Esto permite una descomposición más precisa del alpha y el riesgo."),
        ("🔧", "Opciones y Derivados — Black-Scholes",
         "Instrumentos avanzados",
         "Agregar un módulo de valoración de opciones con el modelo Black-Scholes y griegas (Delta, Gamma, Vega). "
         "Útil para cuberturas y gestión de riesgo con derivados."),
        ("🌍", "Datos Macro en Tiempo Real",
         "Integración FRED / Banco de la República",
         "Conectar la API de FRED (Federal Reserve) para obtener tasas reales, inflación, curva de rendimientos "
         "y otros indicadores macro que contextualicen el riesgo del portafolio en tiempo real."),
        ("🧪", "Stress Testing y Escenarios",
         "Gestión de riesgo avanzada",
         "Simular escenarios adversos específicos: crisis de 2008, COVID-2020, flash crashes. "
         "Evaluar cómo respondería el portafolio bajo esas condiciones extremas."),
        ("📦", "Exportación de Reportes PDF",
         "Productización",
         "Generar reportes automatizados en PDF con gráficos, KPIs y resúmenes ejecutivos. "
         "Útil para presentar resultados a clientes, comités de riesgo o en contextos académicos."),
    ]

    # Mostrar en 2 columnas
    cols = st.columns(2)
    for i, (icon, title, tag, desc) in enumerate(improvements):
        with cols[i % 2]:
            st.markdown(f"""
            <div class='fw-card'>
                <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>
                    <span style='font-size:1.5rem'>{icon}</span>
                    <div>
                        <span class='fw-tag'>{tag}</span><br>
                        <h4 style='margin:0;font-size:0.95rem'>{title}</h4>
                    </div>
                </div>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div class='kpi-card' style='border-top:3px solid {C["pink"]};text-align:center;padding:24px'>
        <div style='font-size:1.8rem'>🏗️</div>
        <h3 style='margin:10px 0 8px;font-size:1rem'>Arquitectura lista para escalar</h3>
        <p style='color:{C["muted"]};font-size:0.88rem;max-width:600px;margin:0 auto'>
            La separación actual en capas (data → logic → API → dashboard) facilita incorporar
            cualquiera de estas mejoras sin reestructurar el proyecto completo.
            El uso de FastAPI con Pydantic y dependencias inyectadas hace que agregar
            nuevos endpoints sea seguro y controlado.
        </p>
    </div>""", unsafe_allow_html=True)

# ===========================================================================
#  FOOTER
# ===========================================================================
st.markdown("---")
st.markdown(
    "<p class='footer'>RiskLab USTA &nbsp;·&nbsp; v2.0 &nbsp;·&nbsp; "
    "Merx Edition &nbsp;·&nbsp; Desarrollado por Antigravity</p>",
    unsafe_allow_html=True,
)
