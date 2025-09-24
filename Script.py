"""
Script para descargar y guardar el dataset Bank Marketing en la carpeta data/.
"""

import os
from ucimlrepo import fetch_ucirepo

os.makedirs("data", exist_ok=True)

  
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 

# Save data
X.to_csv("data/features.csv", index=False)
y.to_csv("data/targets.csv", index=False)

