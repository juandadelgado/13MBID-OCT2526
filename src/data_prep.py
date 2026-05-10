# Se importan las librerías necesarias y se suprimen las advertencias
import pandas as pd
# import numpy as np

def process_data(datos_creditos: str = "data/raw/datos_creditos.csv",
                   datos_tarjetas: str = "data/raw/datos_tarjetas.csv",  
                   output_dir: str = "data/processed/") -> None:

    """ Lee los datos de créditos y tarjetas, realiza el procesamiento. 

    Args:
        datos_creditos (str, optional) : _description_. Default es "data/raw/datos_creditos.csv".
        datos_tarjetas (str, optional) : _description_. Default es "data/raw/datos_tarjetas.csv".
        output_dir (str, optional) : _description_. Default es "data/processed/".
    """

    df_creditos = pd.read_csv("data/raw/datos_creditos.csv", sep=";")
    df_tarjetas = pd.read_csv("data/raw/datos_tarjetas.csv", sep=";")

    #############################################################################
    # Se filtran los datos para eliminar registros con edades superiores a 90 años
    #############################################################################

    df_creditos_filtrado = df_creditos.copy()
    df_creditos_filtrado = df_creditos_filtrado[df_creditos_filtrado['edad'] < 90]

    #############################################################################
    # Tratamiento de valores nulos para tasa de interes y antiguedad_empleado
    #############################################################################
    
    df_creditos_filtrado['tasa_interes'] = df_creditos_filtrado.groupby("objetivo_credito")["tasa_interes"]\
                        .transform(lambda x: x.fillna(x.median()))
    #Tratamiento de nulos para antiguedad_empleado
    df_creditos_filtrado['antiguedad_empleado'] = df_creditos_filtrado.groupby("edad")["antiguedad_empleado"]\
                        .transform(lambda x: x.fillna(x.median()))
    
    #############################################################################
    #Se integran los datos de créditos y tarjetas utilizando el id_cliente como clave
    #############################################################################
    
    df_integrado = pd.merge(df_creditos_filtrado, df_tarjetas, on="id_cliente", how="left")
    
    #############################################################################
    #Se crean nuevos atributos a partir de los datos originales
    #############################################################################

    #Capacidad de pago del cliente
    df_integrado ["capacidad_pago"] = df_integrado["importe_solicitado"] / df_integrado["ingresos"]
    # El número de operaciones mensuales del cliente  
    df_integrado ["operaciones_mensuales"] = df_integrado["operaciones_ult_12m"] / 12
    # Presión financiera del cliente (mensual)
    df_integrado ["presion_financiera"] = (
        (df_integrado["gastos_ult_12m"]/12 + df_integrado["importe_solicitado"]/(df_integrado["duracion_credito"])) 
        / (df_integrado["ingresos"]/12)
    )
    # Gasto promedio por operación realizada
    # Relaciona el gasto total de los últimos 12 meses con el número de operaciones
    df_integrado['gasto_promedio_op'] = df_integrado['gastos_ult_12m'] / df_integrado['operaciones_ult_12m']

    # Cantidad de operaciones mensuales con tarjeta
    # Como los datos de operaciones son de los últimos 12 meses, dividimos por 12
    df_integrado['ops_mensuales_tc'] = df_integrado['operaciones_ult_12m'] / 12

    # Estabilidad laboral
    # Relación entre los años trabajados y la edad del cliente
    df_integrado['estabilidad_laboral'] = df_integrado['antiguedad_empleado'] / df_integrado['edad']
    

    #############################################################################
    #Se eliminan las columnas originales y se integran las nuevas columnas procesadas
    #############################################################################
    
    columnas_a_eliminar = [
        "id_cliente",
        "operaciones_ult_12m",
        "importe_solicitado",
        "duracion_credito",
        "nivel_tarjeta"
    ]


    df_integrado.drop(columnas_a_eliminar, inplace=True, axis=1)

    #############################################################################
    # Se exportan los datos procesados a un nuevo archivo CSV
    #############################################################################
    df_integrado.to_csv('data/processed/datos_integrados.csv', index=False)

if __name__ == "__main__":
    process_data()

