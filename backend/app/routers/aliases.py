"""
Router de alias de endpoints cortos exigidos por la guía del profesor.

La guía pide rutas como `/activos`, `/precios/{ticker}`, `/portafolios`,
`/rendimientos/{ticker}`, `/indicadores/{ticker}`, `/volatilidad/{ticker}`,
`/var`, `/capm`, `/frontera-eficiente`, `/alertas`, `/macro`,
`/curva-rendimiento`, `/bono/duracion`, `/opcion/precio`, `/stress`, `/predict`.

Este módulo expone esos paths cortos como **alias** (proxy) hacia los endpoints
existentes bajo `/api/v1/...` para no romper compatibilidad con el dashboard.
También agrega los endpoints completamente nuevos: `/activos` y `/portafolios`.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.db_models import Asset, Portfolio

router = APIRouter(tags=["alias-corto (guía profesor)"])


# ── /activos ─────────────────────────────────────────────────────────────────

@router.get("/activos", summary="Lista activos disponibles desde BD")
def list_assets(db: Session = Depends(get_db)):
    """Devuelve los activos del catálogo persistido en SQLAlchemy."""
    rows = db.query(Asset).order_by(Asset.ticker.asc()).all()
    return {
        "count": len(rows),
        "assets": [
            {"id": a.id, "ticker": a.ticker, "name": a.name, "sector": a.sector,
             "created_at": a.created_at.isoformat(timespec="seconds") if a.created_at else None}
            for a in rows
        ],
    }


# ── /portafolios (CRUD básico) ───────────────────────────────────────────────

class PortfolioRequest(BaseModel):
    name:    str               = Field(..., min_length=1, max_length=120)
    weights: dict[str, float]  = Field(..., description="Mapa ticker → peso (deben sumar 1).")

    @field_validator("weights")
    @classmethod
    def _weights_sum_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"weights deben sumar 1; suman {total:.4f}")
        for w in v.values():
            if not isinstance(w, (int, float)):
                raise ValueError("Todos los pesos deben ser numéricos")
        return v


@router.get("/portafolios", summary="Lista portafolios persistidos")
def list_portfolios(db: Session = Depends(get_db)):
    rows = db.query(Portfolio).order_by(Portfolio.created_at.desc()).all()
    return {
        "count": len(rows),
        "portfolios": [
            {"id": p.id, "name": p.name, "weights": p.weights,
             "created_at": p.created_at.isoformat(timespec="seconds") if p.created_at else None}
            for p in rows
        ],
    }


@router.post("/portafolios", summary="Crea un portafolio persistido", status_code=201)
def create_portfolio(req: PortfolioRequest, db: Session = Depends(get_db)):
    p = Portfolio(name=req.name, weights=req.weights)
    db.add(p)
    db.commit()
    db.refresh(p)
    return {"id": p.id, "name": p.name, "weights": p.weights,
            "created_at": p.created_at.isoformat(timespec="seconds")}


# ── Alias cortos hacia endpoints existentes ──────────────────────────────────
# Cada alias re-importa el handler original y le pasa los mismos parámetros.
# Esto evita duplicar lógica y mantiene una sola fuente de verdad.

# Para evitar imports circulares, los alias se montan en backend/app/main.py
# después de que todos los handlers principales hayan sido definidos.
# Aquí solo registramos los endpoints CRUD nuevos (/activos, /portafolios).
