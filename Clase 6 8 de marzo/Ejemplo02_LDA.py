# =============================================================================
# Análisis discriminante lineal con estadísticas de Pokémon
# 
# El análisis discriminante lineal es una técnica popular para realizar la reducción de
# dimensionalidad en un conjunto de datos. La reducción de dimensionalidad es la reducción
# de un conjunto de datos de n variables a k variables, donde las variables k son una 
# combinación de las nn variables que conserva o maximiza alguna propiedad útil del 
# conjunto de datos. En el caso del análisis discriminante lineal, las nuevas variables 
# se eligen (y los datos se vuelven a proyectar) de manera que se maximice la separabilidad 
# lineal de un determinado conjunto de clases en los datos subyacentes.
# En otras palabras, dado un conjunto de datos con n variables, incluido un conjunto 
# incrustado de etiquetas que queremos predecir, podemos aplicar LDA a los datos y 
# reducirlo a componentes k, donde esos componentes se eligen de tal manera que maximicen 
# nuestra capacidad de "dibujar líneas" para distinguir las clases.
# Una transformación LDA es útil como un paso de preprocesamiento al modelar 
# clases porque transforma el espacio de tal manera que los algoritmos que luego 
# van y dibujan esos límites, como las máquinas de soporte vectorial, funcionan mucho mejor 
# en los datos transformados que en las proyecciones originales.
# Sin embargo, también es útil como técnica EDA. En esta aplicación, LDA se puede 
# comparar con PCA. PCA es otra técnica de reducción de dimensionalidad 
# (ya vista en clase) que crea nuevas variables que maximizan la varianza del conjunto 
# de datos subyacente. Como tal, funciona en ausencia de etiquetas de datos (es 
# una técnica no supervisada). Mientras tanto, LDA se basa en etiquetas categóricas y 
# crea nuevas variables que maximizan la distinción lineal del conjunto de datos 
# subyacente.
# =============================================================================




# =============================================================================
# Aplicacion 2
# En este segundo ejemplo intentaremos usar LDA para explorar el conjunto de datos 
# de Pokemon. Nuestro objetivo es predecir el tipo de Pokémon basado solo en sus 
# estadísticas totales. 
# =============================================================================



import pandas as pd
pokemon = pd.read_csv('pokemon.csv')
pokemon.head(3)

len(pokemon[pokemon['type2'].isnull()])

# Nos centraremos solo en Pokémon con un solo tipo (por ejemplo, no se permiten 
# tipos duales).

# =============================================================================
# Observe también que estamos normalizando los datos antes de aplicar LDA. 
# La normalización no es estrictamente necesaria, ya que obtendrá el mismo 
# resultado sin ella, solo con números de diferentes tamaños. Aplico la normalización 
# como un paso de preprocesamiento aquí para hacer que los coeficientes de clase 
# sean más grandes (y un poco más fáciles de ver).
df = pokemon[pokemon['type2'].isnull()].loc[
    :, ['sp_attack', 'sp_defense', 'attack', 'defense', 'speed', 'hp', 'type1']
]
X = df.iloc[:, :-1].values

from sklearn.preprocessing import normalize
X_norm = normalize(X)

y = df.iloc[:, -1].values

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(X_norm, y)
# =============================================================================



# =============================================================================
# PCA proporciona un atributo components_ al reductor ajustado, lo que nos permite
# acceder directamente a los componentes del vector. LDA no proporciona este atributo. 
# Esto se debe a que en LDA, la metodología para transformar un vector es un poco más 
# complicada que una simple reproyección w.T * x.
# En cambio, un LDA proporciona un atributo coef_, que es análogo, aunque matemáticamente
# más complicado. Las magnitudes de los componentes en el coef_ nos dicen cuánto 
# pesa cada una de las características hacia la seperabilidad de esa clase.
# Si una clase particular tiene un coeficiente de magnitud particularmente alta 
# (dirección, positiva o negativa, no obstante), entonces esa variable señala 
# muy bien esa clase. Esa variable influirá mucho en la preproyección de LDA. 
# Mientras tanto, un coeficiente de baja magnitud se corresponde con una señal débil 
# y, por lo tanto, se borrará principalmente en la reproyección.
# Si una clase tiene en su mayoría coeficientes de baja magnitud, eso significa que 
# no es fácilmente separable linealmente. Esa clase está relativamente cerca de la 
# media del conjunto de datos o (en los casos más débiles) relativamente cerca de 
# un subconjunto de otras clases en el conjunto de datos.
# VEamos el siguiente mapa de calor : 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

sns.heatmap(pd.DataFrame(lda.coef_, 
                         columns=df.columns[:-1], 
                         index=[lda.classes_]), 
            ax=ax, cmap='RdBu', annot=True)

plt.suptitle('LDA: Coeficientes')
plt.show()
# En este mapa de calor vemos clases que probablemente sean más fáciles de separar, 
# dados sus grandes coeficientes variables, así como clases que probablemente sean
# mucho más difíciles.


# Podemos resumir este mapa de calor mirando los coeficientes totales totales para 
# cada una de las clases. 
pd.Series(np.abs(lda.coef_).sum(axis=1), index=lda.classes_).sort_values().plot.bar(
    figsize=(12, 6), title="LDA Class Coefficient Sums"
)
plt.show()
# Aca observamos, que rock y el ghost son mucho más separables que water y ice.
# Los valores de y tanto en el mapa de calor como en el diagrama de barras pueden 
# tratarse como indicios. Más alto es mejor, pero los números en sí no son 
# particularmente interpretables 
# =============================================================================


# =============================================================================
# Para ver a qué se traduce esta diferencia y comprender qué tan bien nos desempeñamos
# en general, necesitamos pasar a aplicar nuestra LDA.
# Evaluar el rendimiento del clasificador lineal aplicando la proyección LDA
# Para comenzar, como con cualquier técnica de reducción de dimensionalidad, es 
# importante tener en cuenta que cada componente adicional utilizada por el 
# modelo agrega cada vez menos "ganancia" a las reconstrucciones. Por ejemplo, 
# aquí están las tres principales variaciones explicadas de la descomposición de 
# LDA:
lda.explained_variance_ratio_
# Recuerde que PCA selecciona valores que maximizan estos valores directamente. LDA 
# selecciona valores que maximizan las diferencias entre clases, por lo que la 
# explicación de la varianza no se correlacionará exactamente con la utilidad de 
# este o aquel vector en particular. Sin embargo, en la práctica, LDA crea ejes que 
# están razonablemente cerca de los ejes creados por PCA (maximizar la varianza 
# explicada y maximizar la separabilidad de clases son tareas razonablemente similares
# ). Por lo tanto, los puntajes de varianza explicados siguen siendo un indicador 
# útil de medición, y los cortes bruscos en la cantidad de varianza explicada por 
# cada componente son útiles para "cortar" cuántos componentes incluiremos en nuestro 
# conjunto de datos.
X_hat = lda.fit_transform(X, y)

import matplotlib as mpl

colors = mpl.cm.get_cmap(name='tab20').colors
categories = pd.Categorical(pd.Series(y)).categories
ret = pd.DataFrame(
    {'C1': X_hat[:, 0], 'C2': X_hat[:, 1], 'Type': pd.Categorical(pd.Series(y))}
)

fig, ax = plt.subplots(1, figsize=(12, 6))

for col, cat in zip(colors, categories):
    (ret
         .query('Type == @cat')
         .plot.scatter(x='C1', y='C2', color=col, label=cat, ax=ax,
                       s=100, edgecolor='black', linewidth=1,
                       title='Descomposicion LDA: 2 Componentes')
         .legend(bbox_to_anchor=(1.2, 1))
    )
    
    
# Que aprendemos  ... Hemos aprendido que las diferentes clases de Pokémon no 
# se distinguen linealmente por las estadísticas solas.
    
# =============================================================================
# En resumen, este grafico nos dice que clasificar Pokémon usando solo  estadísticas 
# es un problema no lineal. Esto a su vez nos dice que, dado el conjunto actual 
# de características, predecir el tipo de Pokémon es un problema de clasificación 
# muy difícil. La mayoría de los problemas que no son linealmente separables son 
# muy difíciles, a menos que (1) existan dependencias de datos inusualmente 
# complicadas como estructuras espirales o (2) podamos recopilar características 
# adicionales (útiles).
# =============================================================================
# =============================================================================




























































































