from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from src.model_evaluation import evaluation

# ---------------------------
# 1. SCHEMA (entrada)
# ---------------------------
class PredictionInput(BaseModel):
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje: float
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
    tipo_laboral: str
    tendencia_ingresos: str
    tipo_credito: str


class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]


# ---------------------------
# 2. APP
# ---------------------------
app = FastAPI(
    title="Modelo de Riesgo Crediticio",
    description="API para predicción de pago a tiempo",
    version="1.0"
)


# ---------------------------
# 3. CARGA DEL MODELO
# ---------------------------
try:
    model = joblib.load("models/model.joblib")
except Exception as e:
    raise RuntimeError(f"Error cargando modelo: {e}")


# ---------------------------
# 4. ENDPOINTS
# ---------------------------
@app.get("/")
def root():
    return {"message": "API activa 🚀"}


@app.post("/predict")
def predict(batch: BatchPredictionInput):
    try:
        # convertir input a DataFrame
        df = pd.DataFrame([item.dict() for item in batch.data])

        # predicción (pipeline completo)
        preds = model.predict(df)

        return {
            "predictions": preds.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/evaluation")
def get_evaluation():
    try:
        buf = evaluation()
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))