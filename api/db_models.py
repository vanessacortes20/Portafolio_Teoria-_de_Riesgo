"""
Modelos ORM SQLAlchemy para RiskLab USTA.

Convive con la capa sqlite3 directa de api/database.py durante la transición
de Fase 2. Las tablas nuevas se crean con Base.metadata.create_all sobre la
misma base SQLite (data/risklab_users.db) y no interfieren con users ni
reset_tokens existentes.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Asset(Base):
    """Catálogo de activos del portafolio."""
    __tablename__ = "assets"

    id        = Column(Integer, primary_key=True)
    ticker    = Column(String(15), unique=True, nullable=False, index=True)
    name      = Column(String(120))
    sector    = Column(String(60))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    prices = relationship("Price", back_populates="asset", cascade="all, delete-orphan")


class Price(Base):
    """Cache de precios OHLCV descargados de Yahoo Finance."""
    __tablename__ = "prices"

    id       = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    date     = Column(Date, nullable=False, index=True)
    open     = Column(Float)
    high     = Column(Float)
    low      = Column(Float)
    close    = Column(Float)
    volume   = Column(Float)

    asset = relationship("Asset", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("asset_id", "date", name="uq_price_asset_date"),
        Index("ix_prices_asset_date", "asset_id", "date"),
    )


class Portfolio(Base):
    """Portafolios definidos por el usuario o por el sistema."""
    __tablename__ = "portfolios"

    id         = Column(Integer, primary_key=True)
    name       = Column(String(120), nullable=False)
    weights    = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class PredictionLog(Base):
    """Registro de cada predicción del modelo ML servido en /predict (Fase 4)."""
    __tablename__ = "predictions_log"

    id             = Column(Integer, primary_key=True)
    timestamp      = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    model_version  = Column(String(40), nullable=False)
    ticker         = Column(String(15), nullable=False, index=True)
    input_features = Column(JSON)
    prediction     = Column(Float, nullable=False)
    actual         = Column(Float, nullable=True)


class SignalLog(Base):
    """Persistencia de señales técnicas generadas por M7."""
    __tablename__ = "signals_log"

    id        = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ticker    = Column(String(15), nullable=False, index=True)
    rule      = Column(String(40), nullable=False)
    value     = Column(Float)
    note      = Column(String(255))


class FredCache(Base):
    """Cache transparente de series descargadas desde FRED.

    series_id  → identificador FRED (ej: DGS3MO, CPIAUCSL).
    payload    → JSON con la respuesta original de FRED (observations).
    fetched_at → momento de la última descarga; sirve para evaluar TTL.
    """
    __tablename__ = "fred_cache"

    id         = Column(Integer, primary_key=True)
    series_id  = Column(String(40), unique=True, nullable=False, index=True)
    payload    = Column(JSON, nullable=False)
    last_value = Column(Float)
    last_date  = Column(String(10))
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
