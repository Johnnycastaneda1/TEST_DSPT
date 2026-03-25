import io
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.cargar_datos import cargarDatos


def evaluation():

    # ---------------------------
    # 1. CARGAR MODELO Y DATOS
    # ---------------------------
    model = joblib.load("models/model.joblib")

    df = cargarDatos()
    df.columns = [x.replace('__', '_') for x in df.columns]

    X = df.drop(columns=["Pago_atiempo"])
    y = df["Pago_atiempo"]

    # ---------------------------
    # 2. PREDICCIONES
    # ---------------------------
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = None

    # ---------------------------
    # 3. MÉTRICAS
    # ---------------------------
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y, y_proba)

    # ---------------------------
    # 4. CONFUSION MATRIX
    # ---------------------------
    cm = confusion_matrix(y, y_pred)

    # ---------------------------
    # 5. PLOTS
    # ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 📊 métricas
    sns.barplot(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        ax=axes[0]
    )
    axes[0].set_title("Métricas del Modelo")
    axes[0].set_ylim(0, 1)

    # 📊 matriz de confusión
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1]
    )
    axes[1].set_title("Matriz de Confusión")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Real")

    plt.tight_layout()

    # ---------------------------
    # 6. EXPORTAR IMAGEN
    # ---------------------------
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return buf