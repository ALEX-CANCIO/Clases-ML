import numpy as np
import matplotlib.pyplot as plt
xi = np.arange(0,9)
A = np.array([xi,np.ones(9)])
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
w = np.linalg.lstsq(A.T,y)[0]

line = w[0]*xi + w[1] # linea de regresion
plt.plot(xi,line,'r-',xi,y,'o')
plt.show()


# =============================================================================
# Usando Scipy
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scpstats

xi = np.arange(0,9)
A = np.array([ xi, np.ones(9)])
# secuencia generada
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
slope, intercept, r_value, p_value, std_err = scpstats.linregress(xi,y)

print('r value', r_value)
print('p_value', p_value)
print('standard deviation', std_err)

line = slope*xi+intercept
plot(xi,line,'r-',xi,y,'o')
show()


# =============================================================================
# Scikit
# =============================================================================
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import sklearn.model_selection as modelSel
import statsmodels.api


ds.load
boston = ds.load_boston()
boston["DESCR"]
print(boston["DESCR"])
X = boston["data"]
boston["feature_names"]
boston["filename"]
Y = boston["target"]
type(X)
type(Y)
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)
X.columns = boston.feature_names
print(X.head())

X["Price"] = boston.target

Y = X["Price"]
X = X.drop("Price", axis=1)

X_train, X_test, Y_train, Y_test =modelSel.train_test_split(X,
                                                            Y,
                                                            test_size = 0.33,
                                                            random_state = 5)



import sklearn.linear_model as reglin
lm = reglin.LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(Y_test, Y_pred)
plt.xlabel("Precios: $Y_i$")
plt.ylabel("Prediccion de Precios: $\hat{Y}_i$")
plt.title("Precios Vs Precios Predichos: $Y_i$ vs $\hat{Y}_i$")


def f1(Yp,Yt , p):
    return (np.sum((np.abs(Yp - Yt))**p))**(1/p)
Lista1 = []
for p in range(2,501):
    Lista1.append(f1(Y_pred,Y_test,p))
Lista1.index(min(Lista1))
Lista1[212]

def f2(Yp,Yt, q):
    return np.sum( ((Yp-Yt)-q)**2 )/(Yp.shape[0]-1)
q = np.median(Y_pred-Y_test)
q = np.quantile(Y_pred-Y_test,0.55)
q = np.quantile(Y_pred-Y_test,0.45)
f2(Y_pred,Y_test,q)

Lista2= []
Listaq = []
for q  in np.arange(0.01,0.99,0.001):
    q1 =  np.quantile(Y_pred-Y_test,q)
    Lista2.append(f2(Y_pred,Y_test,q1))
    Listaq.append(q)
Lista2
Listaq[Lista2.index(min(Lista2))]
plt.plot(Lista2)
plt.show()





