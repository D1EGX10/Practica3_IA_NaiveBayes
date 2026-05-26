import pandas as pd
import numpy as np

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