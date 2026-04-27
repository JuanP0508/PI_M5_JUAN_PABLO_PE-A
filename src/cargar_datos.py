# Celda 1 - Markdown
## Carga de Datos

# Celda 2 - Código
import os
import pandas as pd

def CargarDatos():
    ruta_actual = os.path.dirname(os.path.abspath("__file__"))
    ruta_excel = os.path.join(ruta_actual, "Base_de_datos.xlsx")
    df = pd.read_excel(ruta_excel)
    return df

df = CargarDatos()

# Celda 3 - Exploración inicial
df.shape  # dimensiones
df.dtypes  # tipos de datos
df.head()  # primeras filas