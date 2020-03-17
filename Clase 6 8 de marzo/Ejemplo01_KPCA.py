# =============================================================================
# Muchos algoritmos de aprendizaje automático hacen suposiciones sobre la 
# separabilidad lineal de los datos de entrada. El perceptrón incluso 
# requiere datos de entrenamiento perfectamente linealmente separables 
# para converger. Otros algoritmos que basicos suponen que la falta 
# de separación lineal perfecta se debe al ruido: Adaline, regresión 
# logística y el SVM (estándar) por nombrar solo algunos. Sin 
# embargo, si estamos lidiando con problemas no lineales, que podemos 
# encontrar con bastante frecuencia en aplicaciones del mundo real, las 
# técnicas de transformación lineal para la reducción de la 
# dimensionalidad, como PCA y LDA, pueden no ser la mejor opción.

# En esta ejemplo, veremos una versión kernelizada de PCA o KPCA. 
# Usando KPCA, aprenderemos cómo transformar datos que no son 
# linealmente separables en un nuevo subespacio de menor dimensión que 
# sea adecuado para clasificadores lineales.
# =============================================================================



# =============================================================================
# Usando algunas funciones auxiliares SciPy y NumPy, veremos que 
# implementar un KPCA es realmente simple:
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np 
def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.    
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_examples, n_features]  
    gamma: float
        Tuning parameter of the RBF kernel    
    n_components: int
        Number of principal components to return    
    Returns
    ------------
    X_pc: {NumPy ndarray}, shape = [n_examples, k_features]
        Projected dataset   
    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')    
    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)    
    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)    
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)    
    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]    
    # Collect the top k eigenvectors (projected examples)
    X_pc = np.column_stack([eigvecs[:, i]
                           for i in range(n_components)])    
    return X_pc

# Una desventaja de usar un RBF para la reducción de dimensionalidad es 
# que tenemos que especificar el parámetro a priori. Encontrar un valor 
# apropiado requiere experimentación y es mejor hacerlo utilizando 
# algoritmos para el ajuste de parámetros, por ejemplo, realizando una 
# grid search.
# =============================================================================


# =============================================================================
# Ejemplo: separación de formas de media luna (half-moon)
from sklearn.datasets import make_moons
X, y = make_moons(100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1],color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1],color='blue', marker='o', alpha=0.5)
plt.tight_layout()
plt.show()
# Para fines de ilustración, la media luna de los símbolos de triángulo 
# representará una clase, y la media luna representada por los símbolos 
# de círculo representará los ejemplos de otra clase:
# =============================================================================


# =============================================================================
# Claramente, estas dos formas de media luna no son linealmente separables
# , y nuestro objetivo es desplegar las medias lunas a través de KPCA 
# para que el conjunto de datos pueda servir como una entrada adecuada 
# para un clasificador lineal. Pero primero, veamos cómo se ve el 
# conjunto de datos si lo proyectamos en los componentes principales a 
# través de PCA estándar:

from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02,color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02,color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()

# Claramente, podemos ver en la figura resultante que un clasificador 
# lineal no podría funcionar bien en el conjunto de datos transformado 
# a través de PCA estándar

# Tenga en cuenta que cuando graficamos solo el primer componente 
# principal (subgrafico derecho), desplazamos los ejemplos triangulares 
# ligeramente hacia arriba y los ejemplos circulares ligeramente hacia 
# abajo para visualizar mejor la superposición de clases. Como muestra 
# el subgrafico izquierdo, las formas de media luna originales solo 
# están ligeramente cortadas y volteadas en el centro vertical: esta 
# transformación no ayudaría a un clasificador lineal a discriminar 
# entre círculos y triángulos. Del mismo modo, los círculos y triángulos 
# correspondientes a las dos formas de media luna no son linealmente 
# separables si proyectamos el conjunto de datos en un eje de 
# características unidimensional, como se muestra en el subgrafico derecho.
# =============================================================================


# =============================================================================
# Ahora, probemos nuestra función de KPCA, rbf_kernel_pca, que 
# implementamos en parrafos anteriores:
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02,color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()
# Ahora podemos ver que las dos clases (círculos y triángulos) están 
# linealmente bien separadas, de modo que tenemos un conjunto de datos 
# de entrenamiento adecuado para clasificadores lineales:

# Desafortunadamente, no hay un valor universal para el parámetro de 
# ajuste, que funcione bien para diferentes conjuntos de datos. 
# Encontrar un valor apropiado para un problema dado requiere experimentación.
# =============================================================================




















































