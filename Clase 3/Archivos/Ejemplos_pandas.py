import pandas as pd
import numpy as np
obj = pd.Series([12,14,190,0.1])

type(obj)

dir(obj)

type(obj.values)

obj.values

obj.values[0]

obj.index

range(1,5)

obj2 = pd.Series([4,6,1,3], index= ['d' , 'b', 'a', 'c'])

%cd "C:\Users\Administrador\Desktop\Clase2" 
%ls


# cargar un archivo de datos 
data = pd.read_csv("FAO.csv",encoding = 'latin-1')

type(data)

type(data['Area Abbreviation'])


data['Y2013'].describe()

data.Y2013[1] = np.nan

data.Y2013.head()


# cargamos uk-500
dataUK = pd.read_csv("uk-500.csv")
dataUK.set_index("last_name", inplace = True)

dataUK.loc['Andrade']

DosPersonas = dataUK.loc[['Andrade','Veness'], 'city':'email']

dataUK.loc[dataUK['first_name'] == "Antonio"]

Hotmail = dataUK.loc[dataUK['email'].str.endswith('hotmail.com')]




# cargamos el dataset titanic
data = pd.read_csv("titanic.csv")
data.head()
data.shape
data.count()


col_names = data.columns.tolist()
for column in col_names:
    print("Valores nulos en <{0}>: {1}".format(column,data[column].isnull().sum()))


data.columns
data.Sex

d = {"male":"M" , "female":"F"}
data["Sex"] = data["Sex"].apply(lambda x:d[x])

pd.crosstab(data.Survived, data.Sex)


%pwd
%ls


opsd_daily = pd.read_csv("opsd_germany_daily.csv")
opsd_daily.columns
type(opsd_daily.Date)

opsd_daily  = opsd_daily.set_index('Date')

opsd_daily.columns
opsd_daily['Year'] = pd.to_datetime(opsd_daily.index).year
opsd_daily['Month'] = pd.to_datetime(opsd_daily.index).month
opsd_daily['Weekday Name'] = pd.to_datetime(opsd_daily.index).weekday

type(opsd_daily.index)

pd.to_datetime(opsd_daily.index).day

opsd_daily.index = pd.to_datetime(opsd_daily.index)
opsd_daily.loc['2012-02']











