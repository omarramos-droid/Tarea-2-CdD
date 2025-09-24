# Tarea-2-CdD
Clasificación supervisada para un dataset


## :open_file_folder: Estructura del proyecto

- `data/` &rarr; lugar de los datasets.
- `reports/` &rarr; figuras y reporte final.
- `main.py` &rarr; script principal.

## Script para los datos
**Install the ucimlrepo package **
pip install ucimlrepo

**Import the dataset into your code **

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# metadata 
print(bank_marketing.metadata) 
  
# variable information 
print(bank_marketing.variables) 
