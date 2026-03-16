# Proyecto de Modelado Predictivo de Créditos

Este repositorio contiene el desarrollo de un **modelo de machine learning** para predecir el comportamiento de nuevos usuarios de crédito en una empresa financiera.  
El proyecto se enmarca dentro de un flujo **MLOps**, garantizando buenas prácticas de colaboración, pruebas, despliegue y monitoreo.

---

## 📂 Estructura del proyecto
```bash
├── mlops_pipeline/
│ └── src/
│ ├── Cargar_datos.ipynb
│ ├── comprension_eda.ipynb
│ ├── ft_engineering.py
│ ├── heuristic_model.py
│ ├── model_training.py
│ ├── model_deploy.py
│ ├── model_evaluation.py
│ └── model_monitoring.py
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
├── readme.md
└── set_up.bat
```

---

## 🚀 Flujo de trabajo

1. **Cargar_datos.ipynb**  
   Notebook para cargar datos desde un CSV de ejemplo.  
   > *En producción, los datos se cargarán desde DWH o Data Lake.*

2. **comprension_eda.ipynb**  
   Análisis exploratorio de datos (EDA) y generación de artefactos experimentales.

3. **ft_engineering.py**  
   Creación de *features* y división de datos en entrenamiento y validación.  
   Incluye construcción de pipelines.

4. **heuristic_model.py**  
   Modelo base de referencia (benchmark heurístico).

5. **model_training.py**  
   Entrenamiento y evaluación de múltiples modelos.  
   - Funciones clave: `summarize_classification` y `build_model`.  
   - Incluye tablas comparativas y visualizaciones.

6. **model_deploy.py**  
   Despliegue del mejor modelo seleccionado.  
   - Creación de una imagen con dependencias y código.  
   - Exposición de un **endpoint** para predicciones batch.

7. **model_evaluation.py**  
   Generación de métricas del modelo desplegado en un dashboard de evaluación.

8. **model_monitoring.py**  
   Monitoreo periódico para:  
   - Detectar *data drift*.  
   - Comparar predicciones vs. datos reales.  
   - Asegurar el desempeño continuo del modelo.

---

## 🚀 Despliegue y Ejecución del Modelo con Docker

Una vez clonado el repositorio se puede levantar el servicio en docker

### 📋 Prerrequisitos

*   **Docker Desktop**: Asegúrate de que Docker esté instalado y en ejecución en tu máquina. Puedes descargarlo desde [el sitio oficial de Docker](https://www.docker.com/products/docker-desktop/).
*   **Git**: Necesitas Git para clonar este repositorio.
*   **Terminal de Comandos**: Una terminal como PowerShell, CMD, Bash, o similar.
*   
### Construcción de imagen con docker
En la terminal ejecutar docker build -t api-creditos .
Una vez construida la imagen, este comando la ejecutará como un contenedor:
docker run -d -p 8000:8000 --name api-creditos-container api-creditos