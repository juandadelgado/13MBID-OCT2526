import streamlit as st
import requests

st.set_page_config(
    page_title="Predicción de mora en créditos",
    page_icon="credit_card:",
    layout="wide"
)


with st.sidebar:
    st.header ("Instrucciones")
    st.write("""
    1. Ingrese los datos del cliente en el formulario.
    2. Haga clic en el botón "Predecir" para obtener la probabilidad de mora en créditos.
    3. Revise los resultados y la información del modelo.         
    """)

    st.divider()
    st.header("Configuración de la API")
    api_url = st.text_input(
        "URL de la API",
        value="http://localhost:8000",
        help="Ingrese la URL donde está alojada la API de predicción."
    )
    st.divider()
    if st.button("Probar conexión con la API"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code ==200:
                st.success("Conexión exitosa a la API.")
            else:
                st.error(f"Error al conectar con la API: {response.status_code}")
        except Exception as e:
            st.error(f"Error al conectar con la API: {e}")

# Título y descripción de la aplicación
st.title("Predicción de mora en créditos")
st.write("Ingrese los datos del cliente para predecir la probabilidad de mora en créditos utilizando nuestro modelo de machine learning entrenado con datos históricos.")

#Formulario de entrada de datos
with st.form("prediction_form"):
    st.subheader("Datos del cliente")
    col1, col2, col3 = st.columns (3)

    with col1:
        edad = st.number_input ("Edad", min_value=18, max_value=100, value=30)
        genero = st.selectbox ("Género", options=["M", "F"])
        estado_civil = st.selectbox ("Estado civil", options=["CASADO", "SOLTERO", "DIVORCIADO", "VIUDO", "OTRO"])

    with col2:
        nivel_educativo = st.selectbox (
            "Nivel educativo", 
            options=["PRIMARIO", "SECUNDARIO", "TERCIARIO", "UNIVERSITARIO_INCOMPLETO", "UNIVERSITARIO_COMPLETO", "POSGRADO"])
        situacion_vivienda = st.selectbox (
            "Situación de vivienda", 
            options=["ALQUILER", "PROPIETARIO", "HIPOTECADA", "OTRA"])
        personas_a_cargo = st.number_input ("Personas a cargo", min_value=0.0, max_value=20.0, value=0.0)

    with col3:
        estado_cliente = st.selectbox ("Estado del cliente", options=["ACTIVO", "INACTIVO"])
        estado_credito = st.selectbox ("Estado del crédito", options=[0, 1])

    st.divider()
    st.subheader("Información financiera y laboral")
    col4, col5, col6 = st.columns (3)

    with col4:
        ingresos = st.number_input ("Ingresos", min_value=0, value=50000)
        antiguedad_empleado = st.number_input ("Antigüedad del empleado (años)", min_value=0, max_value=50, value=1)

    with col5:
        objetivo_credito = st.selectbox (
            "Objetivo del crédito", 
            options=["PERSONAL", "VIVIENDA", "VEHICULO", "NEGOCIOS", "EDUCACION", "OTRO"])
        tasa_interes = st.number_input ("Tasa de interés (%)", min_value=0.0, max_value=100.0, value=10.0)
        pct_ingreso = st.number_input ("Porcentaje de ingreso (%)", min_value=0.0, max_value=1.0, value=0.1)

    with col6:
        capacidad_pago = st.number_input ("Cpacidad de pago", min_value=0.0, value=0.50, step=0.01, format="%.4f")
        presion_financiera = st.number_input ("Presión financiera", min_value=0.0, value=2.0, step=0.01, format="%.4f")
        antiguedad_cliente = st.number_input ("Antigüedad cliente (meses)", min_value=0.0, value=36.0, step=1.0)

    st.divider()
    st.subheader("Gastos y límites") 
    col7, col8 = st.columns (2)

    with col7:
        limite_credito_tc = st.number_input ("Limite de crédito de tarjeta de crédito", min_value=0.0, value=10000.0, step=100.0)

    with col8:
        gastos_ult_12m = st.number_input ("Gastos de los últimos 12 meses", min_value=0.0, value=5000.0, step=100.0,  format="%.4f")
        operaciones_mensuales = st.number_input ("Operaciones mensuales", min_value=0.0, value=5.0, step=1.0)

    st.divider()
    submit_button = st.form_submit_button(
        "Predecir",
        use_container_width=True,
        type="primary" 
    )

if submit_button:
    input_data ={
        "edad": edad,
        "antiguedad_empleado": antiguedad_empleado,
        "situacion_vivienda": situacion_vivienda,
        "ingresos": ingresos,
        "objetivo_credito": objetivo_credito,
        "pct_ingreso": pct_ingreso,
        "tasa_interes": tasa_interes,
        "estado_credito": estado_credito,
        "antiguedad_cliente": antiguedad_cliente,
        "estado_civil": estado_civil,
        "estado_cliente": estado_cliente,
        "gastos_ult_12m": gastos_ult_12m,
        "genero": genero,
        "limite_credito_tc": limite_credito_tc,
        "nivel_educativo": nivel_educativo,
        "personas_a_cargo": personas_a_cargo,
        "capacidad_pago": capacidad_pago,
        "operaciones_mensuales": operaciones_mensuales,
        "presion_financiera": presion_financiera,
    } 

    try:
        response = requests.post(f"{api_url}/predict", json=input_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            st.subheader("Resultados de la predicción")
            st.write(f"**Predicción:** {result['prediction']}")
            st.write("**Probabilidades**")
            for label, prob in result['probability'].items():
                st.write(f"- {label}: {prob:.4f}")
            st.write("**Información del modelo**")
            for key, value in result ['model_info'].items():
                st.write(f"- {key}: {value}")
        else:
            st.error(f"Error en la precicción: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error al conectar a la API: {e}")