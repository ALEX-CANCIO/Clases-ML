import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# =============================================================================
# El análisis de componentes principales es un método no supervisado rápido y flexible 
# para la reducción de la dimensionalidad en los datos
# =============================================================================
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');


# =============================================================================
# A simple vista, está claro que existe una relación casi lineal entre las variables x e y.
# Esto es una reminiscencia de los datos de regresión lineal. Regresión lineal, pero la 
# configuración del problema aquí es ligeramente diferente: en lugar de intentar predecir 
# los valores y a partir de los valores x, el problema de aprendizaje no supervisado 
# intenta aprender sobre la relación entre los valores x e y.
# =============================================================================

# =============================================================================
# En el análisis de componentes principales, esta relación se cuantifica al encontrar 
# una lista de los ejes principales en los datos y al usar esos ejes para describir el 
# conjunto de datos. Usando el estimador PCA de Scikit-Learn, podemos calcular esto de 
# la siguiente manera
# =============================================================================

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

# =============================================================================
# El ajuste aprende algunas cantidades de los datos, lo más importante los "componentes" 
# y la "varianza explicada"
# =============================================================================

print(pca.components_)

print(pca.explained_variance_)

# =============================================================================
# Para ver qué significan estos números, visualicémoslos como vectores sobre los datos 
# de entrada, usando los "componentes" para definir la dirección del vector y la 
# "varianza explicada" para definir la longitud al cuadrado del vector
# =============================================================================

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');


# =============================================================================
# Estos vectores representan los ejes principales de los datos, y la longitud del 
# vector es una indicación de cuán "importante" es ese eje para describir la 
# distribución de los datos; más precisamente, es una medida de la varianza de los 
# datos cuando se proyecta en ese eje. La proyección de cada punto de datos en los 
# ejes principales son los "componentes principales" de los datos.
# =============================================================================

# =============================================================================
# Esta transformación de los ejes de datos a los ejes principales es una transformación 
# afín, lo que básicamente significa que está compuesta por una traslación, rotación 
# y escala.
# =============================================================================

# =============================================================================
# Si bien este algoritmo para encontrar componentes principales puede parecer solo una 
# curiosidad matemática, resulta que tiene aplicaciones de gran alcance en el mundo del 
# aprendizaje automático y la exploración de datos.
# =============================================================================















































































