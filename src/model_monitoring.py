import os
import time
import pandas as pd
import requests
import streamlit as st
st.set_page_config(page_title="Monitoreo del Modelo", layout="wide")
import plotly.express as px
import plotly.graph_objects as go
from evidently.report import Report 
from evidently.metric_preset import DataDriftPreset
from sklearn.model_selection import train_test_split
from src.cargar_datos import cargarDatos
import numpy as np

# ================================
# 1. Configuración
# ================================
API_URL = "http://localhost:8000/predict"
MONITOR_LOG = "monitoring_log.csv"  #dataset para monitorear

# ================================
# 2. Cargar dataset y dividir
# ================================
@st.cache_data
def load_data():
    df = cargarDatos()
    target = "Pago_atiempo"
    X = df.drop(columns=[target])
    y = df[target]
    X_ref, X_new, y_ref, y_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_ref, X_new, y_ref, y_new

X_ref, X_new, y_ref, y_new = load_data()

# ================================
# 3. API para predicciones
# ================================
def get_predictions(X_batch: pd.DataFrame):
    try:
        # 🔥 convertir timestamps a string
        X_batch = X_batch.copy()

        for col in X_batch.select_dtypes(include=["datetime64[ns]"]).columns:
            X_batch[col] = X_batch[col].astype(str)

        X_batch = X_batch.fillna(0)
        X_batch = X_batch.replace([np.inf, -np.inf], 0)

        payload = {
            "data": X_batch.to_dict(orient="records")
        }

        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        return response.json()["predictions"]

    except Exception as e:
        st.error(f"❌ Error conectando con la API: {e}")
        return None

# ================================
# 4. Guardar logs con timestamp
# ================================
def log_predictions(X_batch, preds):
    log_df = X_batch.copy()
    log_df["prediction"] = preds
    log_df["timestamp"] = pd.Timestamp.now()

    if os.path.exists(MONITOR_LOG):
        log_df.to_csv(MONITOR_LOG, mode="a", header=False, index=False)
    else:
        log_df.to_csv(MONITOR_LOG, index=False)

# ================================
# 5. Reporte Evidently
# ================================
def generate_drift_report(ref_data, new_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=new_data)
    return report

# ================================
# 6. Streamlit UI con gráficas
# ================================
st.title("📊 Monitoreo del Modelo en Producción")

# Métricas principales en la parte superior
if os.path.exists(MONITOR_LOG):
    logged_data = pd.read_csv(MONITOR_LOG)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predicciones", len(logged_data))
    with col2:
        st.metric("Predicción Promedio", f"{logged_data['prediction'].mean():.3f}")
    with col3:
        st.metric("Desviación Estándar", f"{logged_data['prediction'].std():.3f}")
    with col4:
        positive_rate = (logged_data['prediction'] > 0.5).mean() * 100
        st.metric("Tasa Positiva (%)", f"{positive_rate:.1f}%")

st.sidebar.header("Opciones")
sample_size = st.sidebar.slider("Tamaño de muestra para monitoreo:", 50, 500, 200)

if st.button("🔄 Generar nuevas predicciones y actualizar log"):
    sample = X_new.sample(n=sample_size, random_state=int(time.time()))
    preds = get_predictions(sample)
    if preds:
        log_predictions(sample, preds)
        st.success("✅ Nuevas predicciones agregadas al log.")
        st.rerun()

# Mostrar datos y gráficas
if os.path.exists(MONITOR_LOG):
    logged_data = pd.read_csv(MONITOR_LOG)
    
    # Crear tabs para organizar mejor
    tab1, tab2, tab3 = st.tabs(["📈 Gráficas", "📊 Data Drift", "📂 Logs"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma de predicciones
            fig_hist = px.histogram(
                logged_data, 
                x='prediction', 
                nbins=20,
                title="Distribución de Predicciones",
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_hist, width="stretch")
        
        with col2:
            # Gráfico de línea temporal (si hay timestamp)
            if 'timestamp' in logged_data.columns:
                logged_data['timestamp'] = pd.to_datetime(logged_data['timestamp'])
                # Agrupar por minuto para mejor visualización
                temporal_data = logged_data.groupby(
                    logged_data['timestamp'].dt.floor('T')
                )['prediction'].mean().reset_index()
                
                fig_time = px.line(
                    temporal_data, 
                    x='timestamp', 
                    y='prediction',
                    title="Evolución Temporal de Predicciones",
                    color_discrete_sequence=['#ff7f0e']
                )
                st.plotly_chart(fig_time, width="stretch")
            else:
                # Box plot como alternativa
                fig_box = px.box(
                    logged_data, 
                    y='prediction',
                    title="Distribución de Predicciones (Box Plot)"
                )
                st.plotly_chart(fig_box, width="stretch")
        
        # Gráfico de comparación con datos de referencia
        st.subheader("🔍 Comparación con Datos de Referencia")
        
        # Seleccionar algunas columnas numéricas para comparar
        numeric_cols = logged_data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'prediction'][:4]  # Solo las primeras 4
        
        if len(numeric_cols) > 0:
            comparison_data = []
            for col in numeric_cols:
                if col in X_ref.columns:
                    comparison_data.append({
                        'Feature': col,
                        'Referencia': X_ref[col].mean(),
                        'Actual': logged_data[col].mean(),
                        'Dataset': 'Comparación'
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name='Referencia',
                    x=comp_df['Feature'],
                    y=comp_df['Referencia'],
                    marker_color='lightblue'
                ))
                fig_comp.add_trace(go.Bar(
                    name='Actual',
                    x=comp_df['Feature'],
                    y=comp_df['Actual'],
                    marker_color='orange'
                ))
                
                fig_comp.update_layout(
                    title="Comparación de Medias: Referencia vs Actual",
                    barmode='group'
                )
                st.plotly_chart(fig_comp, width="stretch")
    
    with tab2:
        st.subheader("📊 Reporte de Data Drift")
        drift_report = generate_drift_report(
            X_ref, logged_data.drop(columns=["prediction", "timestamp"], errors="ignore")
        )

        # Mostrar reporte
        try:
            st.components.v1.html(drift_report._repr_html_(), height=1000, scrolling=True)
        except:
            st.write("✅ Reporte de Drift generado exitosamente")
            st.write(f"📊 Datos de referencia: {X_ref.shape}, Datos actuales: {logged_data.drop(columns=['prediction', 'timestamp'], errors='ignore').shape}")
            
            # Mostrar métricas básicas de drift
            try:
                drift_data = drift_report.as_dict()
                if 'metrics' in drift_data and len(drift_data['metrics']) > 0:
                    dataset_drift = drift_data['metrics'][0].get('result', {}).get('dataset_drift', 'No disponible')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Dataset Drift Detectado", "Sí" if dataset_drift else "No")
                    with col2:
                        # Contar cuántas features tienen drift
                        feature_drifts = drift_data['metrics'][0].get('result', {}).get('drift_by_columns', {})
                        drift_count = sum(1 for v in feature_drifts.values() if v) if feature_drifts else 0
                        st.metric("Features con Drift", f"{drift_count}/{len(feature_drifts)}" if feature_drifts else "0")
            except:
                pass
    
    with tab3:
        st.subheader("📂 Log de Monitoreo")
        
        # Filtro para mostrar más o menos filas
        show_rows = st.selectbox("Mostrar últimas:", [10, 25, 50, 100], index=0)
        st.dataframe(logged_data.tail(show_rows), width="stretch")
        
        # Botón de descarga
        csv = logged_data.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV completo",
            data=csv,
            file_name=f"monitoring_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.warning("⚠️ No hay datos de monitoreo aún. Presiona el botón para iniciar.")