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

        plt.tight_layout()

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

# GAUSSIAN NAIVE BAYES
# IMPLEMENTACION MANUAL

class GaussianNaiveBayes:

    def fit(self, X, y):

        self.clases = np.unique(y)

        self.media = {}

        self.varianza = {}

        self.priori = {}

        for c in self.clases:

            X_c = X[y == c]

            self.media[c] = np.mean(X_c, axis=0)

            self.varianza[c] = np.var(X_c, axis=0)

            self.priori[c] = len(X_c) / len(X)


    def gaussian(self, clase, x):

        media = self.media[clase]

        varianza = self.varianza[clase] + 1e-9

        numerador = np.exp(
            -((x - media) ** 2) / (2 * varianza)
        )

        denominador = np.sqrt(
            2 * np.pi * varianza
        )

        return numerador / denominador


    def predict(self, X):

        predicciones = []

        for x in X:

            probabilidades = []

            for c in self.clases:

                priori = np.log(
                    self.priori[c]
                )

                verosimilitud = np.sum(
                    np.log(
                        self.gaussian(c, x)
                    )
                )

                posterior = priori + verosimilitud

                probabilidades.append(
                    posterior
                )

            predicciones.append(
                self.clases[
                    np.argmax(probabilidades)
                ]
            )

        return np.array(predicciones)

# PRUEBA DEL MODELO

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(
    X_iris.values,
    y_iris.values,
    test_size=0.2,
    random_state=42
)


modelo = GaussianNaiveBayes()

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predicciones
)

print("\n-")
print("RESULTADOS MODELO MANUAL")
print("-")

print(f"Accuracy: {accuracy:.4f}")

# 10-FOLD CROSS VALIDATION

from sklearn.model_selection import KFold


kf = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

accuracies = []


for train_index, test_index in kf.split(X_iris):

    X_train = X_iris.values[train_index]
    X_test = X_iris.values[test_index]

    y_train = y_iris.values[train_index]
    y_test = y_iris.values[test_index]

    modelo = GaussianNaiveBayes()

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predicciones
    )

    accuracies.append(accuracy)


print("\n-")
print("10-FOLD CROSS VALIDATION")
print("-")

print("Accuracies:", accuracies)

print(f"Promedio: {np.mean(accuracies):.4f}")

# LEAVE ONE OUT

from sklearn.model_selection import LeaveOneOut


loo = LeaveOneOut()

accuracies_loo = []


for train_index, test_index in loo.split(X_iris):

    X_train = X_iris.values[train_index]
    X_test = X_iris.values[test_index]

    y_train = y_iris.values[train_index]
    y_test = y_iris.values[test_index]

    modelo = GaussianNaiveBayes()

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predicciones
    )

    accuracies_loo.append(accuracy)


print("\n-")
print("LEAVE ONE OUT")
print("-")

print(f"Promedio: {np.mean(accuracies_loo):.4f}")

# COMPARACION CON SKLEARN

from sklearn.naive_bayes import GaussianNB


X_train, X_test, y_train, y_test = train_test_split(
    X_iris.values,
    y_iris.values,
    test_size=0.2,
    random_state=42
)


modelo_sklearn = GaussianNB()

modelo_sklearn.fit(
    X_train,
    y_train
)

predicciones_sklearn = modelo_sklearn.predict(
    X_test
)

accuracy_sklearn = accuracy_score(
    y_test,
    predicciones_sklearn
)


print("\n-")
print("SKLEARN GAUSSIAN NB")
print("-")

print(f"Accuracy sklearn: {accuracy_sklearn:.4f}")