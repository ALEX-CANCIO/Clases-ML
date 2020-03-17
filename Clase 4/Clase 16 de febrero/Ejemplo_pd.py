%cd "C:\Users\Administrador\Desktop\CLase_6Feb"
%ls
import pandas as pd
import numpy as np
help(np.where)
EducRural = pd.read_csv("indicadoresrural2018.csv",
                        encoding = 'latin1')

type(EducRural)

# Nombre de las columnas
EducRural.columns ==  'RURAL_PMM_MUJDOC'

np.sum(EducRural.columns ==  'RURAL_PMM_MUJDOC')

help("for") 
EducRural.columns.shape

for i in range(0,30):
    bool1 = EducRural.columns[i] ==  'RURAL_PMM_MUJDOC'
    if ( bool1 == True):
         indice = i
print(indice)

def busqueda_Var(val):
    for i in range(0,30):
        bool1 = EducRural.columns[i] ==  val
        if (bool1 == True):
            indice = i
    return indice
        
busqueda_Var("D_DPTO")
EducRural.columns.get_loc('RURAL_PMM_MUJDOC')

np.mean(EducRural.iloc[:,[0,13]].RURAL_PMM_MUJDOC)

# RURAL_CRFA_MUJE1 Estudiantes Mujeres de CRFA entre 12 y 17 años
# RURAL_CRFA_MUJE2 Estudiantes Mujeres de CRFA entre 18 y 20 años
EducRural.columns.get_loc('RURAL_CRFA_MUJE1')
EducRural.columns.get_loc('RURAL_CRFA_MUJE2')

EducRural.iloc[:,[20,21]]









































