# =============================================================================
# PCA como reducción de dimensionalidad
# =============================================================================


# =============================================================================
# El uso de PCA para la reducción de la dimensionalidad implica poner a cero uno o más 
# de los componentes principales más pequeños, lo que resulta en una proyección de menor 
# dimensión de los datos que conserva la máxima varianza de datos.
# =============================================================================

# =============================================================================
# Veamos un ejemplo del uso de PCA como una transformación de reducción de 
# dimensionalidad
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)




# =============================================================================
# Los datos transformados se han reducido a una sola dimensión. Para comprender 
# el efecto de esta reducción de dimensionalidad, podemos realizar la transformación 
# inversa de estos datos reducidos y trazarlos junto con los datos originales
# =============================================================================

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');

# =============================================================================
# Los puntos claros son los datos originales, mientras que los puntos oscuros son la
#  versión proyectada. Esto deja en claro lo que significa una reducción de
#  dimensionalidad de PCA: se elimina la información a lo largo del eje o ejes 
#  principales menos importantes, dejando solo el componente (s) de los datos con 
#  la mayor varianza. La fracción de varianza que se corta (proporcional a la extensión 
# de puntos alrededor de la línea formada en esta figura) es aproximadamente una medida 
# de cuánta "información" se descarta en esta reducción de dimensionalidad.
# =============================================================================

# =============================================================================
# Este conjunto de datos de dimensión reducida es en algunos sentidos 
# "lo suficientemente bueno" para codificar las relaciones más importantes entre 
# los puntos: a pesar de reducir la dimensión de los datos en un 50%, la relación 
# general entre los puntos de datos se conserva principalmente.
# =============================================================================






















































































