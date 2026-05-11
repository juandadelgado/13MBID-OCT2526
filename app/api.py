from fastapi import FastAPI, HTTPException
from matplotlib.pylab import float64
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Dict

app = FastAPI(
    title="Modelo de Prediccón de mora en créditos",
    description="API para predecir la probabilidad de mora en créditos utilizando un modelo de regresión logística entrenado con MLflow.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    edad: int = Field(..., description="Edad del cliente")
    antiguedad_empleado: float = Field(..., description="Antigüedad del empleado")
    situacion_vivienda: str = Field(..., description="Situación de la vivienda")
    ingresos: int = Field(..., description="Ingresos del cliente")
    objetivo_credito: str = Field(..., description="Objetivo del crédito")
    pct_ingreso: float = Field(..., description="Porcentaje de ingreso")
    tasa_interes: float = Field(..., description="Tasa de interés")
    estado_credito: int = Field(..., description="Estado del crédito")
    antiguedad_cliente: float = Field(..., description="Antigüedad del cliente")
    estado_civil: str = Field(..., description="Estado civil")
    estado_cliente: str = Field(..., description="Estado del cliente")
    gastos_ult_12m: float = Field(..., description="Gastos de los últimos 12 meses")
    genero: str = Field(..., description="Género")
    limite_credito_tc: float = Field(..., description="Límite de crédito de la tarjeta de crédito")
    nivel_educativo: str = Field(..., description="Nivel educativo")
    personas_a_cargo: float = Field(..., description="Personas a cargo")
    capacidad_pago: float = Field(..., description="Capacidad de pago")
    operaciones_mensuales: float = Field(..., description="Operaciones mensuales")
    presion_financiera: float = Field(..., description="Presión financiera")
    ops_mensuales_tc: float = Field(..., description="Operaciones mensuales")
    estabilidad_laboral: float = Field(..., description="Estabilidad laboral")
    gasto_promedio_op: float = Field(..., description="Media de gasto")

    class Config:
        json_schema_extra = {
            "example": {
                "edad": 21,
                "antiguedad_empleado": 5.0,
                "situacion_vivienda": "PROPIA",
                "ingresos": 9600,
                "objetivo_credito": "EDUCACIÓN",
                "pct_ingreso": 0.1,
                "tasa_interes": 11.14,
                "estado_credito": 0,
                "antiguedad_cliente": 39.0,
                "estado_civil": "CASADO",
                "estado_cliente": "ACTIVO",
                "gastos_ult_12m": 1144.0,
                "genero": "M",
                "limite_credito_tc": 12691.0,
                "nivel_educativo": "SECUNDARIO_COMPLETO",
                "personas_a_cargo": 3.0,
                "capacidad_pago": 0.104167,
                "operaciones_mensuales": 3.5,
                "presion_financiera": 0.744167,
                "ops_mensuales_tc": 10.5,
                "estabilidad_laboral": 0.23,
                "gasto_promedio_op": 150.0
            }
        }

class PredictionResponse(BaseModel):
    prediction: str 
    probability: Dict[str, float]
    class_labels: Dict[str, str]
    model_info: Dict[str, str]

#Cargar el modelo entrenado
MODEL_PATH = "models/prod_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: no se encontró el modelo en la ruta {MODEL_PATH}. Asegúrate de que el modelo esté en esa ruta")
    model = None
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

@app.get("/")
def read_root():
    return {
        "message": "Bienvenido a la API de Predicción de mora en créditos", 
        "endpoints": { 
            "/predict": "POST - Realiza una predicción de mora en créditos",
            "/docs": "GET - Documentación interactiva de la API",
            "/health": "GET - Verifica el estado de la API"
        }
    }

@app.get("/health")
def health_check():
    if model is not None:
        return {"status": "ok", "message": "La API está funcionando correctamente."}
    else:
        return {"status": "error", "message": "El modelo no está cargado. Verifica el estado del modelo."}
    
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="El modeo no está disponible. Intenta nuevamente.")
    
    try:
        #Convertir la solicitud a un DataFrame
        input_data = pd.DataFrame([request.model_dump()])

        # Realizar la predicción
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # Mapear las etiquetas de clase a descripciones legibles
        class_labels = model.predict(input_data)[0]
        probability_dict = {class_labels[i]: float(probability[i]) for i in range(len(class_labels))}
        model_info = {
            "model_version": "1.0.0",
            "model_type": "Logistic Regression",    
        }

        return PredictionResponse(
            prediction=str(prediction),
            probability=probability_dict,
            class_labels={
                "0": "No entra en mora (N)",
                "1": "Entra en mora (Y)"
            },
            model_info=model_info
        )
    except Exception as e: 
        raise HTTPException(status_code=503, detail=f"Error al realizar la predicción: {e}")
