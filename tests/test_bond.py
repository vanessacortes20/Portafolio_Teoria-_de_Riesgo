"""
Tests unitarios de la clase Bond (modulo M9).

Propiedad clave: cuando YTM = cupon, el bono cotiza al par (precio = face).
"""
from __future__ import annotations

import pytest

from api.services.fixed_income import Bond


def test_bond_at_par_when_ytm_equals_coupon():
    b = Bond(face=1000.0, coupon_rate=0.05, maturity_years=10.0, freq=2)
    price = b.price(ytm=0.05)
    assert abs(price - 1000.0) < 1e-6


def test_bond_above_par_when_ytm_below_coupon():
    b = Bond(face=1000.0, coupon_rate=0.05, maturity_years=10.0, freq=2)
    assert b.price(ytm=0.03) > 1000.0


def test_bond_below_par_when_ytm_above_coupon():
    b = Bond(face=1000.0, coupon_rate=0.05, maturity_years=10.0, freq=2)
    assert b.price(ytm=0.07) < 1000.0


def test_modified_duration_less_than_macaulay():
    b = Bond(face=1000.0, coupon_rate=0.05, maturity_years=10.0, freq=2)
    mac = b.macaulay_duration(0.05)
    mod = b.modified_duration(0.05)
    assert 0.0 < mod < mac


def test_convexity_positive():
    b = Bond(face=1000.0, coupon_rate=0.05, maturity_years=10.0, freq=2)
    assert b.convexity(0.05) > 0


def test_invalid_freq_raises():
    with pytest.raises(ValueError):
        Bond(face=1000.0, coupon_rate=0.05, maturity_years=10.0, freq=3)


def test_invalid_negative_coupon_raises():
    with pytest.raises(ValueError):
        Bond(face=1000.0, coupon_rate=-0.01, maturity_years=10.0, freq=2)
