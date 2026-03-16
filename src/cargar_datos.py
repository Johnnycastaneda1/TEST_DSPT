import os
import pandas as pd

def cargarDatos():
    # ruta del archivo actual (src/cargar_datos.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # subir un nivel (carpeta raíz del proyecto)
    project_root = os.path.dirname(script_dir)
    # construir ruta al archivo Excel
    file_path = os.path.join(project_root, "Base_de_datos.xlsx")
    # leer el archivo
    df = pd.read_excel(file_path)
    return df