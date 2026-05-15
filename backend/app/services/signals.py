"""Generador de señales técnicas (M7).

Encapsula en una clase `SignalGenerator` las cinco reglas exigidas por el
instructivo del Módulo 7:

1. Cruce del MACD (línea MACD cruzando línea de señal).
2. RSI en zonas extremas (umbrales configurables).
3. Bandas de Bollinger (precio tocando banda superior/inferior).
4. Cruce de medias móviles (Golden cross / Death cross).
5. Oscilador Estocástico (%K cruzando %D en zonas extremas).

Cada método devuelve una lista de dicts con campos:
    {id, rule, type, value, msg, explanation}

`evaluate_all()` ejecuta las cinco reglas y devuelve la unión. El resultado
sigue siendo compatible con el response histórico del endpoint
`/api/v1/signals/{ticker}` — las señales nuevas se agregan sin romper las
existentes.
"""
from __future__ import annotations

from typing import Any, Optional


def _f(x: Any, default: float = 0.0) -> float:
    """Coerce a número y reemplaza None/NaN por default."""
    try:
        if x is None:
            return default
        v = float(x)
        if v != v:
            return default
        return v
    except (TypeError, ValueError):
        return default


class SignalGenerator:
    """Genera señales técnicas a partir del último y penúltimo registro técnico.

    Espera dos dicts con campos del M1 (`RSI`, `MACD_Line`, `MACD_Signal`,
    `MACD_Hist`, `Close`, `BB_Upper`, `BB_Lower`, `SMA_20`, `SMA_50`,
    `Stoch_K`, `Stoch_D`).
    """

    def __init__(
        self,
        last: dict,
        prev: dict,
        rsi_overbought: int = 70,
        rsi_oversold:   int = 30,
        stoch_overbought: int = 80,
        stoch_oversold:   int = 20,
    ):
        self.last = last or {}
        self.prev = prev or {}
        self.rsi_overbought = int(rsi_overbought)
        self.rsi_oversold   = int(rsi_oversold)
        self.stoch_overbought = int(stoch_overbought)
        self.stoch_oversold   = int(stoch_oversold)

    # ── Reglas individuales ──────────────────────────────────────────────────

    def macd_cross(self) -> list[dict]:
        """Detecta cruce de la línea MACD respecto a la línea de señal."""
        ml = _f(self.last.get("MACD_Line"))
        sl = _f(self.last.get("MACD_Signal"))
        pml = _f(self.prev.get("MACD_Line"))
        psl = _f(self.prev.get("MACD_Signal"))

        # Si no hay datos de líneas, caemos al histograma (compatibilidad)
        if ml == 0 and sl == 0 and pml == 0 and psl == 0:
            mh  = _f(self.last.get("MACD_Hist"))
            mhp = _f(self.prev.get("MACD_Hist"))
            if mh > 0 and mhp <= 0:
                return [self._signal(
                    "MACD_BULL_CROSS", "MACD cross", "buy", mh,
                    "CRUCE ALCISTA — Histograma MACD positivo",
                    "El histograma MACD acaba de cruzar hacia territorio positivo, lo "
                    "que indica que el momentum de corto plazo supera al de largo "
                    "plazo. Esta señal técnica puede sugerir un arranque alcista, "
                    "pero debe evaluarse junto con otros indicadores y el contexto de riesgo.",
                )]
            if mh < 0 and mhp >= 0:
                return [self._signal(
                    "MACD_BEAR_CROSS", "MACD cross", "sell", mh,
                    "CRUCE BAJISTA — Histograma MACD negativo",
                    "El histograma MACD cruzó hacia territorio negativo. El momentum "
                    "de corto plazo cedió frente al de largo plazo, lo que puede "
                    "anticipar una corrección o desaceleración del precio.",
                )]
            return []

        if ml > sl and pml <= psl:
            return [self._signal(
                "MACD_BULL_CROSS", "MACD cross", "buy", ml - sl,
                f"CRUCE ALCISTA — Línea MACD ({ml:.3f}) cruza por encima de señal ({sl:.3f})",
                "La línea MACD cruzó al alza la línea de señal. Esta es una señal "
                "técnica de momentum positivo de corto plazo, pero no constituye "
                "recomendación; debe confirmarse con otros indicadores y el contexto.",
            )]
        if ml < sl and pml >= psl:
            return [self._signal(
                "MACD_BEAR_CROSS", "MACD cross", "sell", ml - sl,
                f"CRUCE BAJISTA — Línea MACD ({ml:.3f}) cruza por debajo de señal ({sl:.3f})",
                "La línea MACD cruzó a la baja la línea de señal. Puede sugerir "
                "pérdida de momentum, pero no es una recomendación garantizada; "
                "evalúe el contexto de mercado y otros indicadores.",
            )]
        return []

    def rsi_extreme(self) -> list[dict]:
        """Detecta RSI en sobrecompra/sobreventa con umbrales configurables."""
        rsi = _f(self.last.get("RSI"))
        if rsi >= self.rsi_overbought:
            return [self._signal(
                "RSI_OVERBOUGHT", "RSI extreme", "sell", rsi,
                f"SOBRECOMPRA — RSI {rsi:.1f} >= umbral {self.rsi_overbought}",
                f"El RSI ({rsi:.1f}) superó el umbral de sobrecompra ({self.rsi_overbought}). "
                "El activo puede estar sobreextendido en el corto plazo, lo que "
                "históricamente ha coincidido con pausas o retrocesos; no implica "
                "recomendación de venta segura.",
            )]
        if rsi <= self.rsi_oversold and rsi > 0:
            return [self._signal(
                "RSI_OVERSOLD", "RSI extreme", "buy", rsi,
                f"SOBREVENTA — RSI {rsi:.1f} <= umbral {self.rsi_oversold}",
                f"El RSI ({rsi:.1f}) cayó bajo el umbral de sobreventa ({self.rsi_oversold}). "
                "El mercado puede estar siendo excesivamente pesimista con el activo; "
                "esta señal puede sugerir presión vendedora extrema, pero no debe "
                "interpretarse como recomendación de compra garantizada.",
            )]
        return []

    def bollinger_touch(self) -> list[dict]:
        """Detecta precio tocando o cruzando banda superior/inferior de Bollinger."""
        close = _f(self.last.get("Close"))
        upper = _f(self.last.get("BB_Upper"))
        lower = _f(self.last.get("BB_Lower"))
        out: list[dict] = []
        if upper and close >= upper:
            out.append(self._signal(
                "BB_UPPER_TOUCH", "Bollinger touch", "sell", close,
                f"BANDA SUPERIOR — Precio {close:.2f} >= banda superior ({upper:.2f})",
                "El precio tocó o superó la banda superior de Bollinger (~2σ sobre "
                "la media de 20 días). Estadísticamente puede sugerir extensión al alza "
                "y posible reversión a la media; no es recomendación de venta absoluta.",
            ))
        if lower and close <= lower:
            out.append(self._signal(
                "BB_LOWER_TOUCH", "Bollinger touch", "buy", close,
                f"BANDA INFERIOR — Precio {close:.2f} <= banda inferior ({lower:.2f})",
                "El precio tocó o perforó la banda inferior de Bollinger, cotizando "
                "muy por debajo de su rango habitual. Puede sugerir oportunidad si "
                "los fundamentos del activo no han cambiado, pero también puede "
                "anticipar continuación bajista; no es recomendación garantizada.",
            ))
        return out

    def moving_average_cross(self) -> list[dict]:
        """Detecta Golden cross / Death cross entre SMA corta y SMA larga."""
        s_short = _f(self.last.get("SMA_20"))
        s_long  = _f(self.last.get("SMA_50"))
        p_short = _f(self.prev.get("SMA_20"))
        p_long  = _f(self.prev.get("SMA_50"))
        if not s_short or not s_long or not p_short or not p_long:
            return []

        if s_short > s_long and p_short <= p_long:
            return [self._signal(
                "GOLDEN_CROSS", "MA cross", "buy", s_short - s_long,
                f"GOLDEN CROSS — SMA20 ({s_short:.2f}) cruza por encima de SMA50 ({s_long:.2f})",
                "La media móvil corta (SMA20) cruzó por encima de la larga (SMA50). "
                "Este patrón se asocia históricamente con inicios de tendencia "
                "alcista de mediano plazo, pero debe interpretarse junto con "
                "volumen y otros indicadores; no constituye recomendación segura.",
            )]
        if s_short < s_long and p_short >= p_long:
            return [self._signal(
                "DEATH_CROSS", "MA cross", "sell", s_short - s_long,
                f"DEATH CROSS — SMA20 ({s_short:.2f}) cruza por debajo de SMA50 ({s_long:.2f})",
                "La media móvil corta (SMA20) cruzó por debajo de la larga (SMA50). "
                "Patrón asociado con inicio de tendencia bajista de mediano plazo, "
                "aunque puede dar falsas señales; revise volumen y contexto del mercado.",
            )]
        return []

    def stochastic_signal(self) -> list[dict]:
        """Detecta cruces de %K respecto a %D en zonas extremas."""
        k = _f(self.last.get("Stoch_K"))
        d = _f(self.last.get("Stoch_D"))
        pk = _f(self.prev.get("Stoch_K"))
        pd = _f(self.prev.get("Stoch_D"))
        if k == 0 and d == 0 and pk == 0 and pd == 0:
            return []

        if k > d and pk <= pd and k <= self.stoch_oversold:
            return [self._signal(
                "STOCH_BULL", "Stochastic cross", "buy", k,
                f"ESTOCÁSTICO ALCISTA — %K ({k:.1f}) cruza %D ({d:.1f}) en sobreventa",
                f"El %K cruzó al alza al %D estando ambos en zona de sobreventa "
                f"(<= {self.stoch_oversold}). Puede sugerir reversión al alza de corto "
                "plazo, pero el estocástico es ruidoso en mercados con tendencia fuerte; "
                "no constituye recomendación garantizada.",
            )]
        if k < d and pk >= pd and k >= self.stoch_overbought:
            return [self._signal(
                "STOCH_BEAR", "Stochastic cross", "sell", k,
                f"ESTOCÁSTICO BAJISTA — %K ({k:.1f}) cruza %D ({d:.1f}) en sobrecompra",
                f"El %K cruzó a la baja al %D estando ambos en zona de sobrecompra "
                f"(>= {self.stoch_overbought}). Puede sugerir pausa o corrección de "
                "corto plazo; el estocástico puede dar falsas señales, evalúe contexto.",
            )]
        return []

    # ── Agregador ────────────────────────────────────────────────────────────

    def evaluate_all(self) -> list[dict]:
        """Ejecuta las cinco reglas y retorna la lista completa de señales."""
        out: list[dict] = []
        out += self.macd_cross()
        out += self.rsi_extreme()
        out += self.bollinger_touch()
        out += self.moving_average_cross()
        out += self.stochastic_signal()
        return out

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _signal(rule_id: str, rule: str, kind: str, value: float,
                msg: str, explanation: str) -> dict:
        return {
            "id":          rule_id,
            "rule":        rule,
            "type":        kind,
            "value":       float(value) if value is not None else None,
            "msg":         msg,
            "explanation": explanation,
        }
