import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine

# DATASET IRIS
iris = load_iris()

X_iris = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y_iris = pd.Series(iris.target)

print("_IRIS_")
print(X_iris.head())

print("\nClases:")
print(y_iris.unique())

# DATASET WINE

wine = load_wine()

X_wine = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

y_wine = pd.Series(wine.target)

print("\n_WINE_")
print(X_wine.head())

print("\nClases:")
print(y_wine.unique())

# PROBABILIDADES A PRIORI

def probabilidades_priori(y):

    clases = y.unique()

    total = len(y)

    for c in clases:

        cantidad = len(y[y == c])

        probabilidad = cantidad / total

        print(f"Clase {c}")
        print(f"Muestras: {cantidad}")
        print(f"Probabilidad a priori: {probabilidad:.4f}")
        print()


print("\n-")
print("PROBABILIDADES A PRIORI IRIS")
print("-")

probabilidades_priori(y_iris)


print("\n-")
print("PROBABILIDADES A PRIORI WINE")
print("-")

probabilidades_priori(y_wine)

# MEDIA Y DESVIACION ESTANDAR

def estadisticas_por_clase(X, y):

    clases = y.unique()

    for c in clases:

        print("\n-")
        print(f"CLASE {c}")
        print("-")

        datos_clase = X[y == c]

        medias = datos_clase.mean()

        desviaciones = datos_clase.std()

        tabla = pd.DataFrame({
            "Media": medias,
            "Desviacion Estandar": desviaciones
        })

        print(tabla)


print("\n\n-")
print("ESTADISTICAS IRIS")
print("-")

estadisticas_por_clase(X_iris, y_iris)


print("\n\n-")
print("ESTADISTICAS WINE")
print("-")

estadisticas_por_clase(X_wine, y_wine)

# GRAFICAS KDE

def graficas_kde(X, y, nombre_dataset):

    clases = y.unique()

    for columna in X.columns:

        plt.figure(figsize=(8, 5))

        for c in clases:

            sns.kdeplot(
                X[y == c][columna],
                label=f'Clase {c}',
                fill=True
            )

        plt.title(f'KDE - {columna} ({nombre_dataset})')

        plt.xlabel(columna)

        plt.ylabel("Densidad")

        plt.legend()

        plt.grid()

        plt.show()


print("\nGenerando KDE IRIS...")
graficas_kde(X_iris, y_iris, "IRIS")

print("\nGenerando KDE WINE...")
graficas_kde(X_wine, y_wine, "WINE")

# MATRICES DE CORRELACION

def correlacion_por_clase(X, y, nombre_dataset):

    clases = y.unique()

    for c in clases:

        datos_clase = X[y == c]

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            datos_clase.corr(),
            annot=True,
            cmap='coolwarm'
        )

        plt.title(f'Correlacion Clase {c} - {nombre_dataset}')

        plt.tight_layout()

        plt.show()


print("\nGenerando correlaciones IRIS...")
correlacion_por_clase(X_iris, y_iris, "IRIS")

print("\nGenerando correlaciones WINE...")
correlacion_por_clase(X_wine, y_wine, "WINE")