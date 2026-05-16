"""
Modelos ORM del proyecto RiskLab USTA.

Mapean las tablas existentes (users, reset_tokens) y declaran las nuevas
para la capa analitica (assets, prices, portfolios, predictions_log,
signals_log). SQLAlchemy create_all() es idempotente: no recreara las
tablas existentes ni tocara sus datos.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import relationship

from api.db.base import Base


# ── Autenticacion ────────────────────────────────────────────────────────────


class User(Base):
    """Usuario del sistema. Mapea la tabla 'users' creada por la version antigua."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(Text, nullable=False)
    full_name = Column(String(100), nullable=False, default="")
    last_name = Column(String(100), nullable=False, default="")
    phone = Column(String(20), nullable=False, default="")
    cedula = Column(String(20), nullable=True)
    is_active = Column(Integer, default=1)
    role = Column(String(20), default="user")
    created_at = Column(String, nullable=False)
    last_login = Column(String, nullable=True)

    portfolios = relationship(
        "Portfolio", back_populates="user", cascade="all, delete-orphan"
    )
    predictions = relationship("PredictionLog", back_populates="user")

    __table_args__ = (
        Index(
            "idx_users_cedula",
            "cedula",
            unique=True,
            sqlite_where=text("cedula IS NOT NULL"),
        ),
    )


class ResetToken(Base):
    """Token de restablecimiento de contrasena con expiracion."""

    __tablename__ = "reset_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(String, nullable=False)
    used = Column(Integer, default=0)


# ── Datos de mercado ─────────────────────────────────────────────────────────


class Asset(Base):
    """Activo financiero del universo de analisis."""

    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(120), nullable=True)
    sector = Column(String(60), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    prices = relationship(
        "Price", back_populates="asset", cascade="all, delete-orphan"
    )


class Price(Base):
    """Cotizacion OHLCV diaria. Cache transparente de la API externa."""

    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    asset = relationship("Asset", back_populates="prices")

    __table_args__ = (
        Index("ix_prices_asset_date", "asset_id", "date", unique=True),
    )


# ── Portafolios persistidos ──────────────────────────────────────────────────


class Portfolio(Base):
    """Portafolio guardado por un usuario (CRUD /portafolios)."""

    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String(120), nullable=False)
    weights = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="portfolios")


# ── Machine Learning ─────────────────────────────────────────────────────────


class PredictionLog(Base):
    """Registro persistente de cada llamada a /predict."""

    __tablename__ = "predictions_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    model_version = Column(String(40), nullable=False)
    ticker = Column(String(20), nullable=False, index=True)
    input_features = Column(JSON, nullable=False)
    prediction = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=True)
    actual = Column(String(20), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="predictions")


# ── Senales tecnicas ─────────────────────────────────────────────────────────


class SignalLog(Base):
    """Senal tecnica disparada por el sistema (M7)."""

    __tablename__ = "signals_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    rule = Column(String(40), nullable=False)
    signal_type = Column(String(20), nullable=False)
    value = Column(Float, nullable=True)
    message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
