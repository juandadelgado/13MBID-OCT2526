import pandas as pd
from pathlib import Path
import pytest
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*'Number' field should not be instaled.*"
)

import great_expectations as ge

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Number.*should not be instaled.*"),
    pytest.mark.filterwarnings("ignore:.*result_format.*Validation-level.*"),
    pytest.mark.filterwarnings("ignore:.*result_format.*Expectation-level.*")
]

#Paths
PROJECT_DIR =Path (".").resolve()
DATA_DIR = PROJECT_DIR / "data" 

def test_great_expectations():
    """ Prueba para validar la calidad de los datos utilizando Great Expectations.
    """
    # Cargar los datos de créditos y tarjetas
    df_creditos = pd.read_csv(DATA_DIR / "raw/datos_creditos.csv", sep=";")
    df_tarjetas = pd.read_csv(DATA_DIR / "raw/datos_tarjetas.csv", sep=";")

    results = {
        "success": True,
        "expectations": [],
        "statistics": {"success_count": 0, "total_count": 0}    
    }

    def add_expectation(expectation_name, condition, message=""):
        results["statistics"]["total_count"] += 1
        if condition:            
            results["statistics"]["success_count"] += 1
            results["expectations"].append({
            "expectation": expectation_name,
            "success": True
            })
        else:
            results["success"] = False
            results["expectations"].append({
                "expectation": expectation_name,
                "success": False,
                "message": message
            })
    #Atributo a analizar: Exactitud (rangos de valores en datos)
    add_expectation(
        "rango_edad", #Verificar que la edad de los clientes esté entre 18 y 100 años
        df_creditos["edad"].between(18, 100).all(), #La validación a realizar
        "La edad debe estar entre 18 y 100 años." #Mensaje de error en caso de que la validación falle
    )
    add_expectation(
        "situacion_vivienda", #Verificar que la situación de vivienda sea una de las categorías válidas
        df_creditos["situacion_vivienda"].isin(["ALQUILER", "PROPIA", "HIPOTECA", "OTROS"]).all(),
        "La situación de vivienda no se encuentra en el rango válido."
    )
    ############# Validaciones para el dataset de Tarjetas #############

    # Atributo: Exactitud (Rango de valores para límite de crédito)
    # Validamos que el límite de crédito esté en un rango lógico (ej. 0 a 100,000)
    add_expectation(
        "limite_credito_rango", 
        df_tarjetas["limite_credito_tc"].between(0, 100000).all(), 
        "El límite de crédito debe estar entre 0 y 100,000."
    )

    # Atributo: Consistencia/Exactitud (Categorías de nivel educativo)
    # Ajustamos las categorías típicas que suelen venir en estos datasets
    niveles_validos = ["UNIVERSITARIO_COMPLETO", "SECUNDARIO_COMPLETO", "DESCONOCIDO", "UNIVERSITARIO_INCOMPLETO", "POSTGRADO_COMPLETO", "DOCTORADO"]
    add_expectation(
        "nivel_educativo_valido",
        df_tarjetas["nivel_educativo"].isin(niveles_validos).all(),
        f"Existen niveles educativos fuera de las categorías permitidas: {niveles_validos}"
    )