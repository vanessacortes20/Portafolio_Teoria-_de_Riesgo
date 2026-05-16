# Informe Ejecutivo — Proyecto Integrador RiskLab USTA

**Curso:** Teoría del Riesgo · Python para APIs e IA
**Universidad Santo Tomás · Facultad de Estadística · 2024–2026**

> Documento de 5 páginas que resume las decisiones metodológicas, la arquitectura técnica y los resultados clave del sistema. Acompaña al repositorio de código, a la API desplegada y al dashboard interactivo.

---

## 1. Marco conceptual

El **riesgo financiero** es la incertidumbre asociada al valor futuro de un activo o portafolio, expresada en términos cuantificables. La gestión moderna del riesgo lo desagrega en cuatro categorías que el sistema aborda explícita o implícitamente:

- **Riesgo de mercado**: variaciones adversas en precios, tasas, tipos de cambio y volatilidades. Es el dominio principal de RiskLab USTA — los módulos M1 a M6, M9, M10 y M11 lo cuantifican desde ángulos complementarios.
- **Riesgo de crédito**: probabilidad de incumplimiento de la contraparte. Fuera del alcance directo del proyecto, aunque el módulo M9 de renta fija sienta las bases (la duración modificada y la convexidad son insumos del modelado de bonos corporativos sujetos a default).
- **Riesgo de liquidez**: capacidad de cerrar posiciones sin afectar el precio. Implícito en la elección del benchmark líquido (S&P 500) y de activos con histórico diario disponible.
- **Riesgo operativo**: fallas en procesos, sistemas o personas. Atendido por la capa 5 (tests automatizados, contenedor reproducible, CI en cada push).

El **portafolio** se entiende como una combinación lineal de activos individuales cuyos pesos suman uno. La medida central de riesgo de mercado del proyecto es el **Valor en Riesgo (VaR)** — la pérdida máxima esperada a un horizonte y nivel de confianza dados — complementada por el **CVaR** (pérdida promedio condicional dado que se supera el VaR) para capturar el riesgo de cola. El **Expected Shortfall** es coherente en el sentido de Artzner et al. (1999), propiedad que el VaR no tiene.

---

## 2. Decisiones metodológicas

### 2.1 Selección de activos

El portafolio está conformado por **cinco activos** de sectores y geografías diferenciados:

| Ticker | Empresa | Sector | Mercado |
|---|---|---|---|
| `NU` | Nu Holdings | Fintech | Brasil / NYSE |
| `AMZN` | Amazon.com | Consumo / Cloud | EE.UU. |
| `SONY` | Sony Group | Electrónica / Entretenimiento | Japón / NYSE |
| `XOM` | Exxon Mobil | Energía | EE.UU. |
| `WPM` | Wheaton Precious Metals | Materiales / Streaming | Canadá / NYSE |

**Justificación**: cinco sectores no correlacionados positivamente (fintech emergente, tech consolidado, electrónica asiática, energía cíclica, metales preciosos como cobertura inflacionaria). La diversificación sectorial es lo que hace al ejercicio de Markowitz interesante: si los activos estuvieran correlacionados perfectamente, la frontera eficiente se reduciría a un punto.

### 2.2 Modelo de volatilidad

El sistema implementa **dos enfoques complementarios** y los compara:

- **EWMA (RiskMetrics)** con λ = 0.94 por defecto. Recursión cerrada σ²ₜ = λ·σ²ₜ₋₁ + (1−λ)·r²ₜ₋₁. Ventajas: cero parámetros estimados, costo computacional mínimo, decay exponencial claro. Limitaciones: no captura asimetría ni reversión a la media.
- **GARCH(1,1)** con variantes ARCH(1) y EGARCH(1,1). Selección por AIC/BIC. El EGARCH captura el **efecto leverage** (caídas → más volatilidad).

**Conclusión metodológica**: EWMA y GARCH no son sustitutos. EWMA es más adecuado para reporting diario simple; GARCH para escenarios donde importa la estructura paramétrica completa (e.g., pronóstico a varios pasos). El proyecto reporta ambos y deja la decisión al analista.

### 2.3 Método de VaR

El sistema calcula **tres métodos** y los compara:

1. **Paramétrico** asumiendo normalidad — sensible al supuesto, tiende a subestimar pérdidas en colas pesadas.
2. **Histórico** (percentil empírico) — robusto, agnóstico de distribución, requiere histórico suficiente.
3. **Montecarlo** (10 000 simulaciones normales) — flexible, permite incorporar otras distribuciones en el futuro.

El **backtesting de Kupiec POF** se aplica a los tres. Bajo H₀ el modelo está correctamente especificado: LR ~ χ²(1). Si LR > 3.84 al 95%, se rechaza el modelo. El método que pase Kupiec es el que efectivamente describe el riesgo histórico observado del activo.

### 2.4 Optimización por Markowitz

Se formula explícitamente como **problema cuadrático**:

```
minimizar   wᵀ Σ w
sujeto a    Σᵢ wᵢ = 1
            μᵀ w = μ*           (opcional)
            wᵢ ≥ 0               (opcional)
```

Resolución con **cvxpy** (`cp.Minimize(cp.quad_form(w, Σ))`). Se reportan **dos versiones**:

- **Con ventas en corto** (`allow_short = True`): admite pesos negativos. Permite alcanzar rendimientos por encima del activo de mayor μ. Frontera más amplia.
- **Sin ventas en corto** (`allow_short = False`): wᵢ ≥ 0. Realista para inversionistas minoristas. Algunos activos caen a peso cero en la "esquina del simplex" (solución activa de Karush-Kuhn-Tucker).

El **costo de la restricción** se mide explícitamente: para el mismo nivel de Sharpe objetivo, la versión no-short exige más volatilidad. Esta comparación es uno de los aportes pedagógicos del sistema.

### 2.5 Componente de Machine Learning

**Propósito analítico elegido**: clasificación de señal **buy / hold / sell** sobre features técnicas, con etiqueta forward 5 días y umbral ±2%.

**Justificación de la elección**:
- Encaja con el módulo M7 de señales: el ML es una versión generalizada y data-driven de las reglas heurísticas.
- Features ya disponibles en el pipeline existente (RSI, MACD, retornos, volatilidad).
- Métrica de evaluación clara (accuracy, F1 weighted).
- Particionado temporal (`shuffle=False`) evita leakage característico de series financieras.
- El umbral ±2% sobre forward 5 días genera tres clases balanceadas en la mayoría de activos líquidos.

**Modelo**: `Pipeline(StandardScaler + RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced"))`. RF se eligió sobre alternativas (logistic regression, SVM) por su robustez ante features correlacionadas y outliers, y por entregar `predict_proba` que el endpoint expone como `confidence`.

### 2.6 Stress testing

Se aplicaron **seis escenarios** sobre el portafolio Max Sharpe:

| Escenario | Shock | Componente principal |
|---|---|---|
| Tasa +200 pb | Δr = +0.02 | Renta fija + Rf de CAPM |
| Tasa −200 pb | Δr = −0.02 | Renta fija + Rf de CAPM |
| Mercado −20% | ΔR_bench = −0.20 | β-propagación a equities |
| Mercado −30% | ΔR_bench = −0.30 | β-propagación a equities |
| Volatilidad ×2 | σ → 2σ | VaR estresado |
| Tormenta perfecta | +200 pb + −20% + σ×2 | Combinado |

La propagación del shock de mercado usa el β individual estimado por OLS contra el benchmark. La sensibilidad a tasa para equities es opcional (parámetro `equity_rate_duration`, default 0 para evitar dobles cuentas con el efecto market).

---

## 3. Arquitectura técnica

El sistema se organiza en **cinco capas** independientes y testables:

1. **Datos y persistencia** — Yahoo Finance + FRED con cache transparente en SQLite vía SQLAlchemy ORM. Tabla `prices` con índice único `(asset_id, date)` y TTL configurable (24 h por defecto).
2. **Análisis clásico** — `api/logic.py` con indicadores, rendimientos, EWMA, GARCH, CAPM, VaR/CVaR, Kupiec, Markowitz QP. Encapsulado en clases (`OptionPricer`, `Bond`, `YieldCurve`, `StressTester`, `ModelPredictor`).
3. **Renta fija + derivados + stress** — `api/services/{fixed_income,options,stress}.py`. Cada uno expone sus endpoints REST con request/response Pydantic v2 anidados.
4. **Machine Learning** — `api/ml/` con `features.py` (8 features técnicas), `predictor.py` (Singleton verificable), `train.py` (script ejecutable). Artefacto serializado en `model.joblib` + metadata JSON. Logging persistente en `predictions_log`.
5. **Infraestructura** — pytest + TestClient con BD en memoria (StaticPool), Dockerfile multi-stage sobre `python:3.11.9-slim-bookworm`, workflow de GitHub Actions, deploy en Render con disco persistente.

La API expone **22 endpoints REST** documentados automáticamente en `/docs` (Swagger) y `/redoc`. Todos los datos de entrada son validados con Pydantic v2 (campos tipados, `Field()` con descripciones y restricciones, `@field_validator` personalizados). Las respuestas con shape complejo usan **modelos anidados** (`OptionResponse → GreeksModel`, `StressResponse → AssetImpact`, etc.).

---

## 4. Resultados numéricos clave

> *Los siguientes valores se generan al ejecutar el backend contra los datos en vivo. Para reproducir las cifras: arrancar el backend con `uvicorn api.main:app --reload`, abrir el dashboard y presionar "Recalcular (API)", o consultar el endpoint `/api/v1/all`.*

### 4.1 Caracterización estadística del portafolio

Sobre el período 2023-01-01 a hoy, los **rendimientos diarios** del portafolio equiponderado presentan:

- **Asimetría negativa** consistente con el efecto leverage documentado.
- **Curtosis exceso > 0** (colas pesadas) — Jarque-Bera rechaza normalidad en los 5 activos al 1%.
- **Clustering de volatilidad** detectado por Ljung-Box sobre r²ₜ con 10 rezagos (p < 0.05).

Estos hechos justifican el uso de modelos GARCH y de VaR Montecarlo / histórico sobre el paramétrico.

### 4.2 VaR y backtesting

VaR al 95% (diario) por método sobre el portafolio Max Sharpe:

| Método | VaR diario | VaR anualizado | Kupiec |
|---|---|---|---|
| Paramétrico (Normal) | *valor en /var* | × √252 | *LR, p-value* |
| Histórico (percentil) | *valor en /var* | × √252 | *LR, p-value* |
| Montecarlo (10k sims) | *valor en /var* | × √252 | *LR, p-value* |

El método **histórico** suele ser el que pasa Kupiec en activos con colas pesadas, porque no impone supuesto distribucional.

### 4.3 Portafolio óptimo Markowitz

Resuelto por QP con cvxpy:

| Métrica | Sin shorts (`allow_short=False`) | Con shorts (`allow_short=True`) |
|---|---|---|
| Composición | *pesos en /frontier* | *pesos en /frontier* |
| Rendimiento anual | *Return en JSON* | *Return en JSON* |
| Volatilidad anual | *Volatility en JSON* | *Volatility en JSON* |
| Sharpe Ratio | *Sharpe en JSON* | *Sharpe en JSON* |
| Activos en peso 0 | *n_zero_weights* | n/a |

**Costo de la restricción**: la versión sin shorts impone una volatilidad adicional (`extra_volatility_max_sharpe` en `/frontier/compare`). Este valor cuantifica precisamente cuánto cuesta la realidad operativa de no poder vender en corto.

### 4.4 CAPM y benchmark

- **Tasa libre de riesgo**: desde FRED (DGS3MO) en tiempo real, no hardcodeada.
- **Betas individuales** estimados por OLS sobre rendimientos diarios.
- **Alpha de Jensen** del portafolio óptimo contra el S&P 500.
- **Tracking Error** e **Information Ratio** disponibles vía `/api/v1/macro`.

### 4.5 Modelo de Machine Learning

Tras correr `python -m api.ml.train`:

| Métrica | Valor |
|---|---|
| Accuracy (test) | *registrada en .meta.json* |
| F1 (weighted) | *registrada en .meta.json* |
| Tamaño de muestra | *n_train + n_test* |
| Features | 8 (`ret_1d`, `ret_5d`, `ret_20d`, `rsi_14`, `macd_hist`, `vol_20d`, `pos_in_bb`, `volume_ratio`) |
| Etiquetas | Buy / Hold / Sell |

**Observación**: la accuracy en problemas de predicción financiera es difícilmente superior al 60% — si lo fuera, el activo no sería eficiente. El valor del modelo está en la consistencia de las features y en exponer probabilidades calibradas (`confidence`) que el analista puede ponderar con otros insumos.

### 4.6 Stress testing

Pérdidas del portafolio Max Sharpe bajo los seis escenarios (valor base = $100 000):

| Escenario | Pérdida % | Pérdida $ | VaR base 95% | VaR estresado 95% |
|---|---|---|---|---|
| Tasa +200 pb | *cero o pequeña* | — | — | igual al base |
| Tasa −200 pb | *cero o pequeña* | — | — | igual al base |
| Mercado −20% | *~17-22%* | *~17-22 k* | — | igual al base |
| Mercado −30% | *~25-32%* | *~25-32 k* | — | igual al base |
| Volatilidad ×2 | 0% | 0 | — | × 2 aprox |
| Tormenta perfecta | *~20-25%* | *~20-25 k* | — | × 2 aprox |

**Interpretación**: el escenario combinado no es la suma aritmética de los componentes (los efectos no son lineales y σ × 2 no produce pérdida puntual pero sí amplifica el VaR).

---

## 5. Conclusiones y recomendaciones

### Para el inversionista

1. **El portafolio Max Sharpe sin shorts** es el más relevante para inversionistas minoristas. Concentra peso en los activos con mejor relación rendimiento/riesgo histórica, dejando a los demás en cero. Es **sensible al período de estimación**: rebalancear periódicamente.

2. **El VaR paramétrico subestima el riesgo** en colas pesadas. Reportar el VaR histórico o Montecarlo como complemento.

3. **El stress testing** revela que un escenario combinado (caída −20% + volatilidad ×2) puede llevar a una pérdida de 20-25% del capital — magnitud comparable o superior al VaR al 99% estimado bajo supuesto de normalidad. Tener reservas líquidas suficientes para absorber este escenario sin liquidar posiciones forzosamente.

4. **El modelo de ML** debe entenderse como un insumo, no como una decisión. Su valor está en sistematizar la lectura técnica del portafolio en una señal interpretable, no en predecir el mercado.

### Para extensiones futuras

- **Incorporar costos de transacción** y restricciones de tamaño mínimo de posición en el QP de Markowitz.
- **Modelos multifactor** (Fama-French 3 factores, 5 factores) además del CAPM.
- **Reverse stress testing**: encontrar el shock mínimo que produciría pérdida X%.
- **Volatilidad implícita en cadena**: integrar precios de mercado de opciones reales para extraer la superficie de IV.
- **Optimización dinámica**: rebalanceo periódico con costos vs. estática.
- **Monitoreo de drift del modelo ML**: comparar `prediction` vs `actual` en `predictions_log` y reentrenar cuando degrade.

### Para el equipo de operaciones

- La arquitectura permite **horizontalizar**: cambiar SQLite por PostgreSQL solo requiere actualizar `DATABASE_URL` (SQLAlchemy abstrae el dialecto).
- El Singleton del modelo ML es **per-worker**: con múltiples workers de uvicorn, cada uno tiene su propia instancia (acceptable, el modelo se carga 4 veces en lugar de N).
- El **deploy en Render free-tier duerme tras 15 min** sin tráfico. Para producción seria, migrar a un plan pago con instancias siempre activas, o agregar un cron-ping externo.

---

*RiskLab USTA · Proyecto Integrador Teoría del Riesgo · 2024–2026*
*Repositorio: <https://github.com/vanessacortes20/Portafolio_Teoria-_de_Riesgo>*
