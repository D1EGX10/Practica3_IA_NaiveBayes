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

print("===== IRIS =====")
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

print("\n===== WINE =====")
print(X_wine.head())

print("\nClases:")
print(y_wine.unique())