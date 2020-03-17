%pwd

%cd "C:\Users\Administrador\Desktop\Clase1Mar"

# =============================================================================
# Lista de librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

sns.set_palette("deep", desat=0.6)
sns.set_context(rc={"figure.figsize":(8,4)})
# =============================================================================

# =============================================================================
# Cargamos la data en memoria 
Fifa19 = pd.read_csv("datafifa/data.csv",header = 0)
Fifa19.iloc[:5,:5]
# =============================================================================


# =============================================================================
# Info del dataframe Fifa19
Fifa19.columns
Fifa19.Name.count()
Fifa19.Flag.count()
# Verificar la existencia de elementos nulos o vacios
Fifa19.isnull().any().any()
# =============================================================================


# =============================================================================
# Tipos de variables 
# Creamos un objeto de tipo series almacenando
# los nombres de las columnas (variables)
tipos = Fifa19.columns.to_series()

# agrupando
tipos = tipos.groupby(Fifa19.dtypes).groups

tipos[np.dtype('int64')]
tipos[np.dtype('float64')]
tipos[np.dtype('object')]

Fifa19.RS

ctext = tipos[np.dtype('object')]

# Sacando todos los nombres de las columnas 
# definir cnun como una diferencia de conjuntos 
columnas = Fifa19.columns
cnum = list(set(columnas)-set(ctext))
# =============================================================================



# =============================================================================
#   Rellenado de los valores vacios
# Completamos los valores vacios : cnum -> media (mean)
for c in cnum:
    mean = Fifa19[c].mean()
    Fifa19[c] = Fifa19[c].fillna(mean)
    

# Completamos los valores vacios : ctext -> moda
for c in ctext:
    mode = Fifa19[c].mode()[0]
    Fifa19[c] = Fifa19[c].fillna(mode)
    
# Verificamos la existencia de valores na
Fifa19.isnull().any().any()
# =============================================================================


# =============================================================================
# Guardamos en disco duro 
Fifa19.to_csv("datafifa/fifaprocesado.csv", index = False)
# =============================================================================


# =============================================================================
# Estadistica sobre Nationality
Fifa19.Nationality.count()
type(Fifa19.Nationality)
Fifa19.Nationality
Peru = Fifa19[Fifa19.Nationality=="Peru"][["Name","Age","Club","Value","Body Type"]]
Peru.Age.hist().set_title("Edad jugadores Peruanos")
Peru[Peru.Age>35][:]

# Analisis de brasil 
Brazil = Fifa19[Fifa19.Nationality=="Brazil"][["Name","Age","Club","Value","Body Type"]]
Brazil[["Age","Body Type"]].boxplot(by = "Body Type")
# =============================================================================




















