# Practical No. 9
# Title: Implement a single layer perceptron from scratch, train a multi layer perceptron using scikit-learn / keras, compare performance with different activation functions

# !mamba install scikit-learn
# (JupyterLite shell command from the original notebook; commented out because it is not valid in a .py file.
#  Install with: pip install scikit-learn)

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("All Libraries Imported Successfully")

iris = load_iris()

X = iris.data[:100]
y = iris.target[:100]

print("Dataset Loaded Successfully")
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Data Split and Scaled Successfully")

w = np.zeros(X_train.shape[1])
b = 0
lr = 0.1
for epoch in range(100):
    for xi, yi in zip(X_train, y_train):
        pred = 1 if np.dot(xi, w) + b >= 0 else 0
        error = yi - pred
        w += lr * error * xi
        b += lr * error
pred = (np.dot(X_test, w) + b >= 0).astype(int)
print("Single Layer Accuracy:", accuracy_score(y_test, pred))

for act in ['identity', 'logistic', 'tanh', 'relu']:

    model = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation=act,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print(act, "Accuracy:", accuracy_score(y_test, pred))


"""
=================== OUTPUT (text output; plots open in a window when run normally) ===================
All Libraries Imported Successfully
Dataset Loaded Successfully
Features Shape: (100, 4)
Target Shape: (100,)
Data Split and Scaled Successfully
Single Layer Accuracy: 1.0
identity Accuracy: 1.0
logistic Accuracy: 1.0
tanh Accuracy: 1.0
relu Accuracy: 1.0
"""
