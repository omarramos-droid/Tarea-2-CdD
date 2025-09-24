"""
Script para descargar y guardar el dataset Bank Marketing en la carpeta data/.
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Crear carpeta data si no existe
os.makedirs("data", exist_ok=True)

# Descargar dataset Bank Marketing (ID = 222 en UCI)
bank_marketing = fetch_ucirepo(id=222)

# Separar características y etiquetas
X = bank_marketing.data.features
y = bank_marketing.data.targets

# Guardar en la carpeta data
X.to_csv("data/features.csv", index=False)
y.to_csv("data/targets.csv", index=False)

# Guardar metadatos y variables en formato txt para consulta
with open("data/metadata.txt", "w", encoding="utf-8") as f:
    f.write(str(bank_marketing.metadata))

with open("data/variables.txt", "w", encoding="utf-8") as f:
    f.write(str(bank_marketing.variables))

print("✅ Dataset guardado en la carpeta 'data/'")
