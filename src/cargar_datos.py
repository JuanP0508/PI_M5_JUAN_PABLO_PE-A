import os
import pandas as pd

def CargarDatos():
    try:
        ruta_src = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        ruta_src = os.path.join(os.getcwd(), 'src')
    
    ruta_raiz = os.path.dirname(ruta_src)
    ruta_excel = os.path.join(ruta_raiz, "Base_de_datos.xlsx")
    df = pd.read_excel(ruta_excel)
    return df

if __name__ == "__main__":
    datos = CargarDatos()
    print(datos.head())
    print(datos.columns)