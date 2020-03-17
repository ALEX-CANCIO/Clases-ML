import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import sklearn.model_selection as modelSel
import sklearn.linear_model as reglin

boston = ds.load_boston()
X = pd.DataFrame(boston["data"])
Y = pd.DataFrame(boston["target"])
X.columns = boston.feature_names


X_train, X_test, Y_train, Y_test =modelSel.train_test_split(X,
                                                            Y,
                                                            test_size = 0.33,
                                                            random_state = 5)

lm = reglin.LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(Y_test, Y_pred)
plt.xlabel("Precios: $Y_i$")
plt.ylabel("Prediccion de Precios: $\hat{Y}_i$")
plt.title("Precios Vs Precios Predichos: $Y_i$ vs $\hat{Y}_i$")

# =============================================================================
# Calcula la norma-P de vector diferencia entre
# el valor predicho (usando el modelo) : Y_pred
# y el valor de test : Y_test
# =============================================================================

def f1(Yp,Yt):
    Lista1 = []
    for p in range(2,501):
        Lista1.append((np.sum((np.abs(Yp - Yt))**p)).values**(1/p))
    p = Lista1.index(min(Lista1))
    return p
p_run = f1(Y_pred,Y_test)


def f2(Yp,Yt):
    Lista2= []
    Listaq = []
    for q  in np.arange(0.01,0.99,0.001):
        q1 =  np.quantile(Y_pred-Y_test,q)
        Lista2.append(np.sum( ((Yp-Yt)-q1).values**2 )/(Yp.shape[0]-1))
        Listaq.append(q)
    Q = Listaq[Lista2.index(min(Lista2))]
    return Q
Q_run = f2(Y_pred,Y_test)

print("El valor de p que minimiza la norma es %i" %p_run)
print("El percentil que minimiza el indicador es %f" %Q_run)




type((np.sum((np.abs(Y_pred - Y_test))**212))**(1/212))



















