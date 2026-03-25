import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    ShuffleSplit,
    cross_val_score,
    learning_curve
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.ft_engineering import build_preprocessor, _get_feature_columns

# IMPORTANTE: carga de datos
from src.cargar_datos import cargarDatos

# modelo (puedes cambiarlo)
from sklearn.ensemble import RandomForestClassifier


scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def summarize_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def build_model(
    classifier_fn,
    data_params: dict,
    test_frac: float = 0.2,
    save_path_modelo: str = "models/model.joblib",
    save_path_data: str = "models/test_data.joblib"
) -> dict:

    name_of_y_col = data_params["name_of_y_col"]
    names_of_x_cols = data_params["names_of_x_cols"]
    dataset = data_params["dataset"]

    # ---------------------------
    # 1. DATA
    # ---------------------------
    X = dataset[names_of_x_cols].copy() # Riesgo de Leakage
    # 🔥 FIX: asegurar tipos consistentes
    for col in ["tipo_laboral", "tendencia_ingresos", "tipo_credito"]:
        if col in X.columns:
            X[col] = X[col].astype(str)
    Y = dataset[name_of_y_col].copy()

    if "fecha_prestamo" in X.columns:
        X = X.drop(columns=["fecha_prestamo"])

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_frac, random_state=1234
    )

    # ---------------------------
    # 2. PREPROCESSOR
    # ---------------------------
    num, cat, ord_ = _get_feature_columns(x_train)
    preprocessor = build_preprocessor(num, cat, ord_)

    # ---------------------------
    # 3. PIPELINE COMPLETO
    # ---------------------------
    classifier_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier_fn)
        ]
    )

    # ---------------------------
    # 4. TRAIN
    # ---------------------------
    model = classifier_pipe.fit(x_train, y_train)

    # ---------------------------
    # 5. PREDICCIONES
    # ---------------------------
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred)

    # ---------------------------
    # 6. CROSS VALIDATION
    # ---------------------------
    kfold = KFold(n_splits=10)

    cv_results = {}

    for metric in scoring_metrics[:-1]:
        cv_results[metric] = cross_val_score(
            classifier_pipe, x_train, y_train, cv=kfold, scoring=metric
        )

    # ---------------------------
    # 7. LEARNING CURVE
    # ---------------------------
    common_params = {
        "X": x_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=123),
        "n_jobs": -1,
        "return_times": True,
    }

    scoring_metric = "recall"

    train_sizes, train_scores, test_scores, _, _ = learning_curve(
        classifier_pipe, **common_params, scoring=scoring_metric
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, "o-", label="Training score")
    ax.plot(train_sizes, test_mean, "o-", label="Cross-validation score")
    ax.set_title(f"Learning Curve - {classifier_fn.__class__.__name__}")
    ax.legend()
    plt.show()

    # ---------------------------
    # 8. GUARDAR MODELO
    # ---------------------------
    joblib.dump(model, save_path_modelo)
    joblib.dump((x_test, y_test),save_path_data)

    print(f"✅ Modelo guardado en: {save_path_modelo}")
    print(f"✅ Data guardada en: {save_path_data}")

    return {
        "train": train_summary,
        "test": test_summary
    }


# ==========================================
# ENTRYPOINT (AQUÍ SE CARGA EL EXCEL)
# ==========================================
if __name__ == "__main__":

    print("🚀 Iniciando entrenamiento...")

    # ---------------------------
    # 1. CARGAR DATOS
    # ---------------------------
    df = cargarDatos()

    print(f"📊 Datos cargados: {df.shape}")

    # ---------------------------
    # 2. CONFIGURACIÓN
    # ---------------------------
    data_params = {
        "name_of_y_col": "Pago_atiempo",
        "names_of_x_cols": [
            "capital_prestado",
            "plazo_meses",
            "edad_cliente",
            "salario_cliente",
            "total_otros_prestamos",
            "cuota_pactada",
            "puntaje",
            "puntaje_datacredito",
            "cant_creditosvigentes",
            "huella_consulta",
            "saldo_mora",
            "saldo_total",
            "saldo_principal",
            "saldo_mora_codeudor",
            "creditos_sectorFinanciero",
            "creditos_sectorCooperativo",
            "creditos_sectorReal",
            "promedio_ingresos_datacredito",
            "tipo_laboral",
            "tendencia_ingresos",
            "tipo_credito"
        ],
        "dataset": df
    }

    # ---------------------------
    # 3. ENTRENAR
    # ---------------------------
    result = build_model(
        classifier_fn=RandomForestClassifier(),
        data_params=data_params
    )

    print("✅ Entrenamiento terminado")
    print(result)