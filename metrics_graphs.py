import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Lista de rutas a CSV junto con el nombre del modelo
rutas = [
    (Path("metrics_model25") / "resultados_metricas.csv", "Conditional 25%"),
    (Path("metrics_model50") / "resultados_metricas.csv", "Conditional 50%"),
    (Path("metrics_model75") / "resultados_metricas.csv", "Conditional 75%"),
    (Path("metrics_model100") / "resultados_metricas.csv", "Conditional 100%")
]

# Métricas a graficar
metricas = ['SSIM', 'PSNR', 'MSE']

# Diccionario donde guardar los DataFrames por modelo
dataframes = {}

# Leer y almacenar cada CSV
for ruta, nombre in rutas:
    if not ruta.exists():
        print(f"❌ Archivo no encontrado: {ruta}")
        continue
    df = pd.read_csv(ruta)
    dataframes[nombre] = df

# Verificar que haya al menos un archivo cargado
if not dataframes:
    raise ValueError("No se pudo cargar ningún CSV. Verifica las rutas.")

# Obtener clases
primer_nombre = list(dataframes.keys())[0]
clases = sorted(dataframes[primer_nombre]['Clase'].unique())

# Estilos de marcadores por modelo
marcadores = ['o', 's', '^', 'D', '*', 'x', 'P', 'v']  # Se repiten si hay más modelos
colores = plt.cm.tab10.colors  # Paleta por defecto

# Generar gráfico de líneas con marcadores para cada métrica
for metrica in metricas:
    plt.figure(figsize=(12, 6))
    valores_maximos = []
    valores_minimos = []

    for i, (nombre_modelo, df) in enumerate(dataframes.items()):
        df_sorted = df.sort_values("Clase")
        valores = df_sorted[metrica].values
        plt.plot(clases, valores,
                 marker=marcadores[i % len(marcadores)],
                 color=colores[i % len(colores)],
                 label=nombre_modelo,
                 linewidth=2)

        valores_maximos.extend(valores)
        valores_minimos.extend(valores)

    # Ajuste eje Y con precisión
    ymin = min(valores_minimos)
    ymax = max(valores_maximos)
    margen = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
    plt.ylim(ymin - margen, ymax + margen)

    plt.title(f'Comparison of {metrica} by class')
    plt.xlabel('Class')
    plt.ylabel(metrica)
    plt.xticks(clases)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Leyenda fuera del gráfico
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.12), ncol=len(dataframes))

    plt.tight_layout()
    plt.savefig(f'lineas_{metrica}.png')
    plt.show()
