%pwd

%cd "C:\Users\Administrador\Desktop\Clase1Mar"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

sns.set_palette("deep", desat=0.6)
sns.set_context(rc={"figure.figsize":(8,4)})

ONG_data = pd.read_csv("data/Entrenamiento.csv",header = 0)

ONG_data.iloc[:5,:5]

# Dos formas para acceder a cada una de las variables
ONG_data["DONOR_AMOUNT"].count()
ONG_data.DONOR_AMOUNT.count()


# Verifiquemos la existencia de elementos nulos o vacios
ONG_data.isnull().any().any()


# Visualizar los nombres de las variables
type(ONG_data.columns)
ONG_data.columns

# Creamos un objeto de tipo series almacenando
# los nombres de las columnas (variables)
tipos = ONG_data.columns.to_series()

# agrupando
tipos = tipos.groupby(ONG_data.dtypes).groups

# Accediendo a los valores que tienen 
# la llave np.dtype('int64')
tipos[np.dtype('int64')]

# Lista de tipos de datos (vairiables)
# no numericas
ctext = tipos[np.dtype('object')]
len(ctext)


# Sacando todos los nombres de las columnas 
# definir cnun como una diferencia de conjuntos 
columnas = ONG_data.columns
cnum = list(set(columnas)-set(ctext))
len(cnum)


# Completamos los valores vacios : cnum -> media (mean)
for c in cnum:
    mean = ONG_data[c].mean()
    ONG_data[c] = ONG_data[c].fillna(mean)
    

# Completamos los valores vacios : ctext -> moda
for c in ctext:
    mode = ONG_data[c].mode()[0]
    ONG_data[c] = ONG_data[c].fillna(mode)
    
# Verificamos la existencia de valores na
ONG_data.isnull().any().any()

# Almacenamos la data limpia en disco duro
ONG_data.to_csv("data/entrenamiento_procesado.csv",index = False)

porcent_donantes = (ONG_data[ONG_data.DONOR_AMOUNT>0]['DONOR_AMOUNT'].count()*
                    1.0/ONG_data['DONOR_AMOUNT'].count())*100

porcent_donantes


# grafico de tipo pie
donantes = ONG_data.groupby('DONOR_FLAG').IDX.count()
labels = ['Donante\n'+str(round(x*1.0/donantes.sum()*100,2))
          + "%" 
          for x in donantes]
labels[0] = 'NO' + labels[0]

plt.pie(donantes,labels= labels)
plt.title("Proporcion de donantes")
plt.savefig("data/pie_donantes.png")


ONG_donantes = ONG_data[ONG_data.DONOR_AMOUNT > 0]

len(ONG_donantes)    

imp_segm = pd.cut(ONG_donantes['DONOR_AMOUNT'], 
                  [0,10,20,30,40,50,60,100,200])

plot = pd.value_counts(imp_segm).plot(kind = 'bar',
                                      title ="Importes [Donacion]")
plot.set_ylabel("Cantidad de Donantes")
plot.set_xlabel("Rango de Importes")
plt.savefig("data/Histograma_donantes.png")

pd.value_counts(imp_segm)




# =============================================================================

imp_segm = pd.cut(ONG_donantes['DONOR_AMOUNT'], 
                  [0,10,20,30,40,50,200])

plot = pd.value_counts(imp_segm).plot(kind = 'bar',
                                      title ="Importes [Donacion]")
plot.set_ylabel("Cantidad de Donantes")
plot.set_xlabel("Rango de Importes")
plt.savefig("data/Histograma_donantes.png")

pd.value_counts(imp_segm)
# =============================================================================

# Donacion promedio
ONG_donantes['DONOR_AMOUNT'].mean()



# Diagrama de cajas
sns.boxplot(list(ONG_donantes['DONOR_AMOUNT']))
plt.title("importe de DOnacion")
plt.savefig("data/Boxplot_donacion.png")


# =============================================================================
# Analisis por genero
# =============================================================================
ONG_donantes.groupby('GENDER').size().plot(kind = "bar")

ONG_donantes[(ONG_donantes.DONOR_AMOUNT <=50) & 
             (ONG_donantes.GENDER.isin(['F','M']))][['DONOR_AMOUNT','GENDER']].boxplot(by = 'GENDER')

























































