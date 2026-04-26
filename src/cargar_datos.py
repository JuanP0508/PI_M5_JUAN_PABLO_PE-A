import os 
import pandas as pd 
def CargarDatos():

    ruta_actual=os.path.dirname(os.path.abspath(__file__))

    ruta_proyecto=os.path.dirname(ruta_actual)

    ruta_excel=os.path.join(ruta_proyecto,"Base_de_datos.xlsx")

    df=pd.read_excel(ruta_excel)
    print(df)
    return df


if __name__ == "__main__":
    datos = CargarDatos()
    print(datos.head())
    print(datos.columns)