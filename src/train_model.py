import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
)
from mlflow.models import infer_signature
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Configura Matplotlib para no usar un backend gráfico
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) 

def load_data(data_path: str = "data/processed/datos_integrados.csv"):
    """Carga los datos desde un archivo CSV.
    
    Args:
        data_path (str): Ruta al archivo CSV con los datos procesados.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """

    df = pd.read_csv(data_path)

    # Se divide el dataset en variables predictoras y variable objetivo
    target = "falta_pago"

    features_X = df.drop(columns=[target])
    labels_y = df[target]

    # Se genera el conjunto de entrenamiento, validación y test con estratificación
    # Primero separar test final (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_X,
        labels_y,
        test_size=0.10,
        random_state=42,
        stratify=labels_y
    )

    # Luego separar train y validation (22% del 90% es aprox. el 20% del total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.22,
        random_state=42,
        stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, y_train, y_val, y_test, features_X, labels_y

def create_preprocessor (features_X: pd.DataFrame):
    """Crea un preprocesador para las variables numéricas y categóricas.
    
    Args:
        features_X (pd.DataFrame): DataFrame con las variables predictoras.

    Returns:
        ColumnTransformer: Preprocesador para las variables numéricas y categóricas.
    """

   # Se identifican las columnas numéricas y categóricas

    num_cols = features_X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = features_X.select_dtypes(include=["object","category"]).columns.tolist()
    
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])
    return preprocessor

def train_model(
        data_path: str = "data/processed/datos_integrados.csv",
        model_path: str = "models/prod_model.pkl",
        preprocessor_path: str = "models/prod_preprocessor.pkl",
        metrics_path: str = "metrics/train_metrics.json",
):
    """Entrena un modelo para predecir la variable objetivo.
    
    Args:
        data_path (str): Ruta al archivo CSV con los datos procesados.
        model_path (str): Ruta donde se guardará el modelo entrenado.
        preprocessor_path (str): Ruta donde se guardará el preprocesador.
        metrics_path (str): Ruta donde se guardarán las métricas de entrenamiento.

    Returns:
        None
    """

    # Configurar MLFlow  
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Proyecto 13MBID - Modelo de Predicción de Falta de Pago")

    #Cargar los datos
    X_train, y_train, X_val, y_val, X_test, y_test, y_train, y_val, y_test, features_X, labels_y = load_data(data_path)

    #Se codifica la variable objetivo para Mlflow 
    
    if set(labels_y.dropna().unique()) == {"N", "Y"}:
        y_train_eval = y_train.map({"N": 0, "Y": 1})
        y_test_eval = y_test.map({"N": 0, "Y": 1})
    else:
        y_train_eval = y_train.copy()
        y_test_eval = y_test.copy()

    # Preparación del Pipeline con Logistic Regression y Random Under Sampling
    modelo = LogisticRegression(max_iter=2000)
    
    preprocessor = create_preprocessor(features_X)
    
    pipeline = ImbPipeline([
            ("prep", preprocessor),
            ("undersample", RandomUnderSampler(random_state=42)),
            ("model", modelo)
        ])
    
    #Entrenamiento del modelo
    pipeline.fit(X_train, y_train_eval)

    #Evaluación del modelo
    y_test_pred = pipeline.predict(X_test)
    y_test_score = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else pipeline.decision_function(X_test)
    )

    metrics = {
        "test_accuracy": accuracy_score(y_test_eval, y_test_pred),
        "test_precision": precision_score(y_test_eval, y_test_pred, zero_division=0),
        "test_recall": recall_score(y_test_eval, y_test_pred, zero_division=0),
        "test_f1": f1_score(y_test_eval, y_test_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test_eval, y_test_score)
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Se genera una matriz de confusión para el conjunto de test

    cm = confusion_matrix(y_test_eval, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión")
    cm_path = "docs/figures/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # Registro en MLflow

    signature = infer_signature(X_train, pipeline.predict(X_train))


    with mlflow.start_run(run_name=f"Pipeline (prod) - LogisticRegression"):
        mlflow.log_params(modelo.get_params())
        mlflow.log_params({
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "balancing_method": "undersampling",
        })

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(
            pipeline, 
            artifact_path="model", 
            signature=signature
        )

        run_id = mlflow.active_run().info.run_id
        print(f"Modelo registrado en MLflow. run_id: {run_id}")

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, model_path)
        joblib.dump(pipeline.named_steps["prep"], preprocessor_path)

        #Guardar métricas
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return pipeline, pipeline.named_steps["prep"], metrics

if __name__ == "__main__":
    train_model()
        

#############################################################
# TODO: Escoger el mejor modelo según nuestra experimentación
# test_accuracy: 0.8549
# test_precision: 0.5543
# test_recall: 0.8644
# test_f1: 0.6755
# test_roc_auc: 0.9203
#############################################################
# EXTRA: en la fase previa, trabajando en la libreta una opción de mejora es utilizar 
# una técnica como GridSearchCV o RandomizedSearchCV para optimizar los hiperparámetros del modelo seleccionado. 
#############################################################