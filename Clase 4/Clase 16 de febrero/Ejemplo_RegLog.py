import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import sklearn.linear_model as reg_lin
dataframe = pd.read_csv("usuarios_win_mac_lin.csv")
dataframe.head()
dataframe.describe()

print(dataframe.groupby('clase').size())


dataframe.drop(['clase'],1).hist()
plt.show()

sb.pairplot(dataframe.dropna(), hue='clase',size=4,
            vars=["duracion", "paginas","acciones","valor"],kind='reg')

#construyamos los vectores de datos 
X = np.array(dataframe.drop(['clase'],1))
y = np.array(dataframe['clase'])


model = reg_lin.LogisticRegression()
model.fit(X,y)
predictions = model.predict(X)
model.score(X,y)







































