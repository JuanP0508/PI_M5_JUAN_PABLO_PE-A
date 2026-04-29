import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="API Predicción Pago a Tiempo", version="1.0")

# Cargar modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelo = joblib.load(os.path.join(BASE_DIR, "mejor_modelo.pkl"))

# Schema de entrada
class CreditoInput(BaseModel):
    tipo_credito: int
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str

class BatchInput(BaseModel):
    registros: List[CreditoInput]

@app.get("/")
def root():
    return {"mensaje": "API de predicción de pago a tiempo - PI_M5 Juan Pablo"}

@app.post("/predict")
def predict(data: BatchInput):
    df = pd.DataFrame([r.dict() for r in data.registros])
    predicciones = modelo.predict(df).tolist()
    probabilidades = modelo.predict_proba(df)[:, 1].tolist()
    return {
        "predicciones": predicciones,
        "probabilidad_pago_atiempo": [round(p, 4) for p in probabilidades],
        "total_registros": len(predicciones)
    }