import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Monitor Data Drift - Créditos", layout="wide")

# ── Ruta al archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_EXCEL = os.path.join(os.path.dirname(BASE_DIR), "Base_de_datos.xlsx")

# ── Variables a monitorear
NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "puntaje_datacredito", "saldo_mora"
]
CATEGORICAS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]

# ──────────────────────────────────────────────
# FUNCIONES DE DRIFT
# ──────────────────────────────────────────────

def calcular_psi(referencia, actual, bins=10):
    breakpoints = np.linspace(
        min(referencia.min(), actual.min()),
        max(referencia.max(), actual.max()),
        bins + 1
    )
    ref_counts, _ = np.histogram(referencia, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)
    ref_pct = ref_counts / len(referencia)
    act_pct = act_counts / len(actual)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)
    psi = np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))
    return round(psi, 4)

def calcular_js(referencia, actual, bins=30):
    minv = min(referencia.min(), actual.min())
    maxv = max(referencia.max(), actual.max())
    h_ref, _ = np.histogram(referencia, bins=bins, range=(minv, maxv), density=True)
    h_act, _ = np.histogram(actual,    bins=bins, range=(minv, maxv), density=True)
    h_ref = (h_ref + 1e-10) / (h_ref + 1e-10).sum()
    h_act = (h_act + 1e-10) / (h_act + 1e-10).sum()
    return round(float(jensenshannon(h_ref, h_act)), 4)

def semaforo(valor, metrica):
    umbrales = {
        "psi":           (0.1, 0.2),
        "ks":            (0.1, 0.2),
        "js":            (0.05, 0.1),
        "chi2_p":        (0.05, 0.01),  # invertido: menor p = más drift
    }
    bajo, alto = umbrales[metrica]
    if metrica == "chi2_p":
        if valor > bajo:  return "🟢 Estable"
        elif valor > alto: return "🟡 Alerta"
        else:              return "🔴 Crítico"
    else:
        if valor < bajo:  return "🟢 Estable"
        elif valor < alto: return "🟡 Alerta"
        else:              return "🔴 Crítico"

# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────

st.title("📊 Monitor de Data Drift — Modelo de Crédito")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_excel(RUTA_EXCEL)
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])
    return df.sort_values("fecha_prestamo").reset_index(drop=True)

df = cargar_datos()

# ── Sidebar: configurar ventanas de tiempo
st.sidebar.header("Configuración")
fecha_min = df["fecha_prestamo"].min().date()
fecha_max = df["fecha_prestamo"].max().date()

corte = st.sidebar.date_input("Fecha de corte (fin referencia / inicio actual)",
    value=pd.to_datetime("2025-06-01").date(),
    min_value=fecha_min, max_value=fecha_max
)

df_ref = df[df["fecha_prestamo"].dt.date < corte]
df_act = df[df["fecha_prestamo"].dt.date >= corte]

st.sidebar.metric("Registros referencia", len(df_ref))
st.sidebar.metric("Registros actuales",   len(df_act))

if len(df_ref) < 10 or len(df_act) < 10:
    st.warning("Muy pocos datos en alguna ventana. Cambia la fecha de corte.")
    st.stop()

# ── Calcular métricas numéricas
resultados = []
for col in NUMERICAS:
    r = df_ref[col].dropna()
    a = df_act[col].dropna()
    if len(r) < 5 or len(a) < 5:
        continue
    ks_stat, ks_p = stats.ks_2samp(r, a)
    psi_val = calcular_psi(r.values, a.values)
    js_val  = calcular_js(r.values, a.values)
    resultados.append({
        "Variable":     col,
        "KS Statistic": round(ks_stat, 4),
        "KS p-value":   round(ks_p, 4),
        "PSI":          psi_val,
        "JS Divergence":js_val,
        "Estado":       semaforo(psi_val, "psi"),
    })

df_metricas = pd.DataFrame(resultados)

# ── Calcular métricas categóricas
resultados_cat = []
for col in CATEGORICAS:
    r = df_ref[col].dropna().astype(str)
    a = df_act[col].dropna().astype(str)
    cats = set(r.unique()) | set(a.unique())
    tabla = pd.DataFrame({
        "ref": r.value_counts().reindex(cats, fill_value=0),
        "act": a.value_counts().reindex(cats, fill_value=0)
    })
    chi2, p = stats.chi2_contingency(tabla)[:2]
    resultados_cat.append({
        "Variable": col,
        "Chi2":     round(chi2, 4),
        "p-value":  round(p, 4),
        "Estado":   semaforo(p, "chi2_p"),
    })

df_cat = pd.DataFrame(resultados_cat)

# ──────────────────────────────────────────────
# SECCIÓN 1: Resumen con semáforo
# ──────────────────────────────────────────────
st.subheader("🚦 Estado General del Modelo")

criticos = (df_metricas["Estado"] == "🔴 Crítico").sum() + (df_cat["Estado"] == "🔴 Crítico").sum()
alertas  = (df_metricas["Estado"] == "🟡 Alerta").sum()  + (df_cat["Estado"] == "🟡 Alerta").sum()
estables = len(df_metricas) + len(df_cat) - criticos - alertas

c1, c2, c3 = st.columns(3)
c1.metric("🟢 Estables",  estables)
c2.metric("🟡 En Alerta", alertas)
c3.metric("🔴 Críticas",  criticos)

# Recomendación automática
if criticos > 0:
    vars_criticas = df_metricas[df_metricas["Estado"] == "🔴 Crítico"]["Variable"].tolist()
    st.error(f"⚠️ Se recomienda **reentrenar el modelo**. Variables con drift crítico: {', '.join(vars_criticas)}")
elif alertas > 0:
    st.warning("🔍 Algunas variables muestran señales de drift. Monitorear de cerca.")
else:
    st.success("✅ El modelo opera sobre una población estable. No se requieren acciones.")

# ──────────────────────────────────────────────
# SECCIÓN 2: Tablas de métricas
# ──────────────────────────────────────────────
st.subheader("📋 Métricas de Drift — Variables Numéricas")
st.dataframe(df_metricas, use_container_width=True, hide_index=True)

st.subheader("📋 Métricas de Drift — Variables Categóricas")
st.dataframe(df_cat, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# SECCIÓN 3: Gráfico de barras de riesgo (PSI)
# ──────────────────────────────────────────────
st.subheader("📊 Indicador de Riesgo por Variable (PSI)")

fig = px.bar(
    df_metricas.sort_values("PSI", ascending=False),
    x="Variable", y="PSI", color="PSI",
    color_continuous_scale=["green", "yellow", "red"],
    range_color=[0, 0.3],
    title="PSI por variable | Alerta > 0.1 | Crítico > 0.2"
)
fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Alerta (0.1)")
fig.add_hline(y=0.2, line_dash="dash", line_color="red",    annotation_text="Crítico (0.2)")
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# SECCIÓN 4: Comparación distribuciones
# ──────────────────────────────────────────────
st.subheader("🔍 Distribución Histórica vs Actual")

variable = st.selectbox("Selecciona una variable:", NUMERICAS)

fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=df_ref[variable].dropna(), name="Referencia",
    opacity=0.6, marker_color="steelblue", nbinsx=30))
fig2.add_trace(go.Histogram(x=df_act[variable].dropna(), name="Actual",
    opacity=0.6, marker_color="tomato", nbinsx=30))
fig2.update_layout(barmode="overlay", title=f"Distribución: {variable}")
st.plotly_chart(fig2, use_container_width=True)

# ──────────────────────────────────────────────
# SECCIÓN 5: Evolución temporal del drift
# ──────────────────────────────────────────────
st.subheader("📈 Evolución Temporal del Drift (KS Statistic)")

var_temporal = st.selectbox("Variable para análisis temporal:", NUMERICAS, key="temp")
ref_vals = df_ref[var_temporal].dropna().values

registros_temp = []
df_ord = df.sort_values("fecha_prestamo")
fechas = pd.date_range(df_ord["fecha_prestamo"].min(), df_ord["fecha_prestamo"].max(), freq="30D")

for fecha in fechas[1:]:
    ventana = df_ord[
        (df_ord["fecha_prestamo"] >= fecha - pd.Timedelta(days=30)) &
        (df_ord["fecha_prestamo"] < fecha)
    ][var_temporal].dropna().values
    if len(ventana) >= 5:
        ks, _ = stats.ks_2samp(ref_vals, ventana)
        registros_temp.append({"Fecha": fecha, "KS": round(ks, 4)})

if registros_temp:
    df_temp = pd.DataFrame(registros_temp)
    fig3 = px.line(df_temp, x="Fecha", y="KS", markers=True,
        title=f"KS Statistic a lo largo del tiempo — {var_temporal}")
    fig3.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Alerta")
    fig3.add_hline(y=0.2, line_dash="dash", line_color="red",    annotation_text="Crítico")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No hay suficientes datos para el análisis temporal.")

st.caption("PI_M5 — Juan Pablo | Monitor de Data Drift")