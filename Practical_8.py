# Practical No. 8
# Title: Train a linear / curve SVM, margins & support vectors, tune C & gamma hyperparameter

# !mamba install scikit-learn
# (JupyterLite shell command from the original notebook; commented out because it is not valid in a .py file.
#  Install with: pip install scikit-learn)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

print("All Libraries Imported Successfully")

iris = load_iris()

X = iris.data[:100, :2]
y = iris.target[:100]

print("Dataset Loaded Successfully")
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

sc = StandardScaler()

X = sc.fit_transform(X)

print("Data Scaled Successfully")

models = {
    'Linear': SVC(kernel='linear', C=1),
    'RBF': SVC(kernel='rbf', C=1, gamma=0.5),
    'Polynomial': SVC(kernel='poly', C=1, degree=3)
}

print("SVM Models Created Successfully")

for name, model in models.items():

    model.fit(X, y)

    plt.figure()

    plt.scatter(X[:, 0], X[:, 1], c=y)

    ax = plt.gca()

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 100),
        np.linspace(ylim[0], ylim[1], 100)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, levels=[0.5], colors='black')

    plt.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        facecolors='none',
        edgecolors='red'
    )

    plt.title(name + ' SVM')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

for C in [0.1, 1, 10]:

    model = SVC(kernel='rbf', C=C, gamma=0.5)

    model.fit(X, y)

    print("C =", C, "Accuracy =", model.score(X, y))

for gamma in [0.1, 0.5, 1]:

    model = SVC(kernel='rbf', C=1, gamma=gamma)

    model.fit(X, y)

    print("Gamma =", gamma, "Accuracy =", model.score(X, y))


"""
=================== OUTPUT (text output; plots open in a window when run normally) ===================
All Libraries Imported Successfully
Dataset Loaded Successfully
Features Shape: (100, 2)
Target Shape: (100,)
Data Scaled Successfully
SVM Models Created Successfully
C = 0.1 Accuracy = 0.99
C = 1 Accuracy = 0.99
C = 10 Accuracy = 1.0
Gamma = 0.1 Accuracy = 0.99
Gamma = 0.5 Accuracy = 0.99
Gamma = 1 Accuracy = 1.0
"""
