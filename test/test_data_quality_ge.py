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
    # Validación 1: Rango de edad (18 a 100 años)
    edad_valida = df_creditos["edad"].between(18, 100).all()
    mensaje_edad = ""
    if not edad_valida:
        edades_fuera = df_creditos[(df_creditos["edad"] < 18) | (df_creditos["edad"] > 100)]["edad"].unique()
        mensaje_edad = f"Existen edades fuera del rango encontradas: {sorted(edades_fuera)}"
    add_expectation(
        "rango_edad", #Verificar que la edad de los clientes esté entre 18 y 100 años
        edad_valida, #La validación a realizar
        f"La edad debe estar entre 18 y 100 años. {mensaje_edad}"
    )

    # Validación 2: Situación de vivienda (ALQUILER, PROPIA, HIPOTECA, OTROS)
    vivienda_valida = df_creditos["situacion_vivienda"].isin(["ALQUILER", "PROPIA", "HIPOTECA", "OTROS"]).all()
    mensaje_vivienda = ""
    if not vivienda_valida:
        viviendas_fuera = df_creditos[~df_creditos["situacion_vivienda"].isin(["ALQUILER", "PROPIA", "HIPOTECA", "OTROS"])]["situacion_vivienda"].unique()
        mensaje_vivienda = f"Existen situaciones de vivienda fuera del rango encontradas: {sorted(viviendas_fuera)}"
    add_expectation(
        "situacion_vivienda", #Verificar que la situación de vivienda sea una de las categorías válidas
        vivienda_valida,
        f"La situación de vivienda no se encuentra en el rango válido. {mensaje_vivienda}"
    )

# Validación 3: Límite de crédito (0 a 100,000)
    limite_valido = df_tarjetas["limite_credito_tc"].between(0, 100000).all()
    mensaje_limite = ""
    if not limite_valido:
        limites_fuera = df_tarjetas[(df_tarjetas["limite_credito_tc"] < 0) | (df_tarjetas["limite_credito_tc"] > 100000)]["limite_credito_tc"].unique()
        mensaje_limite = f"Límites de crédito fuera de rango encontrados: {sorted(limites_fuera)}"
    add_expectation(
        "limite_credito_rango", 
        limite_valido, 
        f"El límite de crédito debe estar entre 0 y 100,000. {mensaje_limite}"
    )

    # Validación 4: Nivel educativo (Categorías específicas)
    niveles_validos = ["UNIVERSITARIO_COMPLETO", "SECUNDARIO_COMPLETO", "DESCONOCIDO", "UNIVERSITARIO_INCOMPLETO", "POSGRADO_COMPLETO", "POSGRADO_INCOMPLETO", "DOCTORADO"]
    educacion_valida = df_tarjetas["nivel_educativo"].isin(niveles_validos).all()
    mensaje_educacion = ""
    if not educacion_valida:
        educacion_fuera = df_tarjetas[~df_tarjetas["nivel_educativo"].isin(niveles_validos)]["nivel_educativo"].unique()
        mensaje_educacion = f"Niveles educativos no reconocidos encontrados: {sorted(educacion_fuera)}"
    add_expectation(
        "nivel_educativo_valido",
        educacion_valida,
        f"Existen niveles educativos fuera de las categorías permitidas. {mensaje_educacion}"
    )

    # Resumen y validación final
    print("n" + "+"*70)
    print("RESUMEN DE VALIDACIONES")
    print("+"*70)
    for exp in results["expectations"]:
        status = "✅" if exp["success"] else "❌"
        print(f"{status} {exp['expectation']}")
        if not exp["success"] and "message" in exp:
            print(f"   Detalles: {exp['message']}")
    print(f"\nTotal: {results['statistics']['success_count']}/{results['statistics']['total_count']}, Éxitos: {results['statistics']['success_count']}, Fallos: {results['statistics']['total_count'] - results['statistics']['success_count']}")
    print("="*70 + "\n")

    # El test falla si alguna validación no se cumple
    assert results["success"], f"Se encontraron {results['statistics']['total_count'] - results['statistics']['success_count']} validaciones fallidas. Revisa el resumen para más detalles."
    
