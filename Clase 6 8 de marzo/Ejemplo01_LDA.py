%cd /media/abraham/DATA/Documentos/Progra4DataScience_CTIC/LDA

# =============================================================================
# El análisis discriminante lineal (LDA) es una técnica de reducción de dimensionalidad.
# Como su nombre lo indica, las técnicas de reducción de dimensionalidad reducen el número
# de dimensiones (es decir, las variables) en un conjunto de datos mientras retienen la 
# mayor cantidad de información posible.
# =============================================================================

# =============================================================================
# Veamos cómo podríamos implementar el análisis discriminante lineal desde cero utilizando Python. Para comenzar, 
# importemos las siguientes bibliotecas que usaremos:
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# =============================================================================






# =============================================================================
# En el ejemplo siguiente, trabajaremos con el conjunto de datos wine que se puede 
# obtener del repositorio de aprendizaje automático UCI. Afortunadamente, la biblioteca
# scitkit-learn proporciona una función para descargar:
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Categorical.from_codes(wine.target, wine.target_names)
# =============================================================================


# El conjunto de datos contiene 178 filas de 13 columnas cada una.
X.shape


# Las variables se componen de varias características, como el contenido de magnesio 
# y alcohol del vino.
X.head()


# Hay 3 tipos diferentes de vino.
wine.target_names


# Creamos un DataFrame que contiene tanto las características como las clases.
df = X.join(pd.Series(y, name='class'))


# =============================================================================
# El análisis discriminante lineal se puede dividir en los siguientes pasos:
# 1. Calcular las matrices de dispersión dentro de las clases y entre clases
# 2. Calcule los vectores propios y los valores propios correspondientes para las matrices de dispersión 
# 3. Ordene los valores propios y seleccione los k superiores
# 4. Cree una nueva matriz que contenga vectores propios que se asignen a los k valores propios
# 5. Obtenga las nuevas características (es decir, los componentes LDA) tomando el producto punto de los datos y la matriz del paso 4
# =============================================================================

# =============================================================================
# Dentro de la matriz de dispersión de clase

# Para cada clase, creamos un vector con las medias de cada caracteristica.
class_feature_means = pd.DataFrame(columns=wine.target_names)
for c, rows in df.groupby('class'):
    class_feature_means[c] = rows.mean()
class_feature_means


# Luego, conectamos los vectores medios (mi) en la ecuación anterior para obtener la
# matriz de dispersión dentro de la clase.
within_class_scatter_matrix = np.zeros((13,13))
for c, rows in df.groupby('class'):
    rows = rows.drop(['class'], axis=1)
    s = np.zeros((13,13))
    for index, row in rows.iterrows():
        x, mc = row.values.reshape(13,1), class_feature_means[c].values.reshape(13,1)
        s += (x - mc).dot((x - mc).T)
    within_class_scatter_matrix += s
# =============================================================================
    
# =============================================================================
# Matriz de dispersion entre clases 
feature_means = df.mean()
between_class_scatter_matrix = np.zeros((13,13))
for c in class_feature_means:    
    n = len(df.loc[df['class'] == c].index)
    mc, m = class_feature_means[c].values.reshape(13,1), feature_means.values.reshape(13,1)
    between_class_scatter_matrix += n * (mc - m).dot((mc - m).T)
# =============================================================================

# =============================================================================
# Luego, resolvemos el problema del valor propio generalizado para 
# obtener los discriminantes lineales.
eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))
# =============================================================================




# =============================================================================
# Los vectores propios con los valores propios más altos llevan la mayor cantidad de 
# información sobre la distribución de los datos. Por lo tanto, clasificamos los valores 
# propios de mayor a menor y seleccionamos los primeros k vectores propios. Con el fin 
# de garantizar que el valor propio se asigne al mismo vector propio después de la 
# clasificación, los colocamos en una matriz temporal.
pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
for pair in pairs:
    print(pair[0])
# =============================================================================

# =============================================================================
# Simplemente observando los valores, es difícil determinar qué parte de la varianza 
# se explica por cada componente. Por lo tanto, lo expresamos como un porcentaje.
eigen_value_sums = sum(eigen_values)
print('Explained Variance')
for i, pair in enumerate(pairs):
    print('Eigenvector {}: {}'.format(i, (pair[0]/eigen_value_sums).real))
# =============================================================================



# =============================================================================
# Primero, creamos una matriz W con los dos primeros vectores propios.
w_matrix = np.hstack((pairs[0][1].reshape(13,1), pairs[1][1].reshape(13,1))).real

# Luego, guardamos el producto escalar de X y W en una nueva matriz Y.
X_lda = np.array(X.dot(w_matrix))
# =============================================================================






# =============================================================================
# matplotlib no puede manejar variables categóricas directamente. Por lo tanto, 
# codificamos cada clase como un número para poder incorporar las etiquetas de clase 
# en nuestro gráfico.
le = LabelEncoder()
y = le.fit_transform(df['class'])

# Luego, graficamos los datos en función de los dos componentes LDA y usamos un 
# color diferente para cada clase.
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_lda[:,0],
    X_lda[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)
plt.show()
# =============================================================================

# =============================================================================
# En lugar de implementar el algoritmo Linear Discriminant Analysis desde cero cada vez,
 # podemos usar la clase predefinida LinearDiscriminantAnalysis que nos ofrece la 
# biblioteca scikit-learn.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)

# Podemos acceder al metodo explained_variance_ratio_ para obtener la varianza explicada
# por cada componente.
lda.explained_variance_ratio_

# Al igual que antes, trazamos los dos componentes LDA.
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_lda[:,0],
    X_lda[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)
# =============================================================================





# =============================================================================
# A continuación, echemos un vistazo a cómo se compara LDA con el análisis de 
# componentes principales o PCA. Comenzamos creando y ajustando una instancia de la 
# clase PCA.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X, y)

# Podemos acceder a la propiedad shown_variance_ratio_ para ver el porcentaje de 
# la varianza explicada por cada componente.
pca.explained_variance_ratio_

# Como podemos ver, PCA seleccionó los componentes que darían lugar a la mayor 
# difusión (retener la mayor cantidad de información) y no necesariamente los que 
# maximizan la separación entre clases.

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)
# =============================================================================

# =============================================================================
# A continuación, veamos si podemos crear un modelo para clasificar el uso de los 
# componentes LDA como características. Primero, dividimos los datos en conjuntos 
# de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, random_state=1)


# Luego, construimos y entrenamos un árbol de decisión. Después de predecir la 
# categoría de cada muestra en el conjunto de pruebas, creamos una matriz de
# confusión para evaluar el rendimiento del modelo.
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
confusion_matrix(y_test, y_pred)
# Como podemos ver, el clasificador del Árbol de decisión clasificó correctamente 
# todo en el conjunto de prueba.
# =============================================================================











