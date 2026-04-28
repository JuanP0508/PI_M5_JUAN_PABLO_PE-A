# PI_M5 — Modelo de Predicción de Pago a Tiempo

**Juan Pablo Pena Gutierrez** | Data Science

---

## Caso de Negocio

Una cooperativa financiera necesita predecir si un cliente pagará su crédito a tiempo (`Pago_atiempo`), basándose en variables financieras, demográficas y de historial crediticio. El objetivo es reducir la mora y mejorar la asignación de crédito mediante un modelo de clasificación supervisado, complementado con un sistema de monitoreo continuo que detecte cambios en la población.

---

## Dataset

| Característica | Detalle |
|---|---|
| Registros | ~1.000+ créditos |
| Período | Nov 2024 – Ene 2026 |
| Target | `Pago_atiempo` (binario: 1 = pagó a tiempo, 0 = no pagó) |
| Variables numéricas | capital_prestado, puntaje, salario_cliente, saldo_mora, entre otras |
| Variables categóricas | tipo_credito, tipo_laboral, tendencia_ingresos |

---

## Estructura del Proyecto

```
PI_M5_JUAN_PABLO_PE-A/
├── Base_de_datos.xlsx                   # Dataset original
├── README.md                            # Documentación del proyecto
└── src/
    ├── cargar_datos.py                  
    ├── comprension_eda.ipynb            
    ├── ft_engineering.py                
    ├── model_training_evaluation.py     
    └── model_monitoring.py              
|-- requirements

---

## Avances

### Avance 1 — Exploración y Preprocesamiento
- Carga y validación del dataset (`cargar_datos.py`)
- Análisis exploratorio de datos: distribuciones, correlaciones y valores atípicos (`comprension_eda.ipynb`)
- Transformación de variables y feature engineering (`ft_engineering.py`)
- Encoding de variables categóricas y tratamiento de nulos

### Avance 2 — Modelamiento y Evaluación
- Entrenamiento de modelo de clasificación binaria (`model_training_evaluation.py`)
- Evaluación con métricas: Accuracy, F1-Score, AUC-ROC
- Selección de variables por importancia
- Generación de predicciones sobre el conjunto de prueba

### Avance 3 — Monitoreo de Data Drift
- Implementación de métricas estadísticas de drift (`model_monitoring.py`):
  - KS Test (Kolmogorov-Smirnov) para variables numéricas
  - PSI (Population Stability Index)
  - Jensen-Shannon Divergence
  - Chi-cuadrado para variables categóricas
- Aplicación Streamlit con dashboard interactivo
- Sistema de alertas automáticas por semáforo (🟢 🟡 🔴)
- Análisis de evolución temporal del drift

---

## Principales Hallazgos

- partimos de una base de datos muy desbalanceada quizas por la aleatoriedad de sus datos, se realizaron correcciones y balanceo

- Las variables 'salario_cliente' y 'capital_prestado' son las más relevantes para predecir el pago a tiempo.

- Se detectó mayor concentración de clientes 'Independientes' en períodos recientes, lo que genera drift en 'tipo_laboral'.

- El PSI de 'saldo_mora' presentó variabilidad temporal, indicando sensibilidad estacional en el comportamiento de los clientes.

- El modelo requiere monitoreo periódico cada 30 días para garantizar la estabilidad de sus predicciones.

- La variable puntaje fue eliminada durante el preprocesamiento por presentar riesgo de data leakage: al ser un score calculado internamente con información del comportamiento de pago, su inclusión inflaría artificialmente las métricas del modelo sin aportar capacidad predictiva real en producción.

---

## Ejecución de la App de Monitoreo

```bash
# Instalar dependencias
pip install streamlit pandas numpy scipy plotly openpyxl

# Ejecutar la app
streamlit run model_monitoring.py
```

---

## Umbrales de Alerta

| Métrica | 🟡 Alerta | 🔴 Crítico |
|---|---|---|
| KS Statistic | > 0.10 | > 0.20 |
| PSI | > 0.10 | > 0.20 |
| JS Divergence | > 0.05 | > 0.10 |
| Chi² p-value | < 0.05 | < 0.01 |

---

## Tecnologías Utilizadas

- Python 3.x
- Pandas, NumPy, Scipy
- Scikit-learn
- Streamlit
- Plotly
- Git / GitHub
