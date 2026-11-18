# Practical No. 7
# Title: Train logistic regression for binary classification, visualize the sigmoid function & decision boundary, analyze ROC curve

import numpy as np
import matplotlib.pyplot as plt

print("Libraries Imported Successfully")

np.random.seed(42)
n_samples = 200
X = np.vstack([
    np.random.randn(100, 2) + [-2, -2],
    np.random.randn(100, 2) + [2, 2]
])
y = np.array([0] * 100 + [1] * 100)
indices = np.random.permutation(n_samples)

X = X[indices]
y = y[indices]

print("Dataset Created Successfully")
print("X Shape:", X.shape)
print("Y Shape:", y.shape)

indices = np.random.RandomState(42).permutation(len(X))
test_size = 40
test_indices = indices[:test_size]
train_indices = indices[test_size:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
w = np.zeros(X_train.shape[1])
b = 0

lr = 0.1
epochs = 3000
for i in range(epochs):
    z = np.dot(X_train, w) + b
    y_output = sigmoid(z)
    error = y_output - y_train
    dw = np.dot(X_train.T, error) / len(y_train)
    db = np.mean(error)
    w -= lr * dw
    b -= lr * db
print("Logistic Regression Model Trained Successfully")
y_prob = sigmoid(np.dot(X_test, w) + b)
y_pred = (y_prob >= 0.5).astype(int)
accuracy = np.mean(y_test == y_pred)
print("Accuracy:", accuracy * 100, "%")

x_values = np.linspace(-10, 10, 100)
y_values = sigmoid(x_values)
plt.figure(figsize=(6, 4))
plt.plot(x_values, y_values)
plt.title("Sigmoid Function")
plt.xlabel("Input")
plt.ylabel("Probability")
plt.grid()
plt.show()

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)
Z = sigmoid(w[0] * xx.ravel() + w[1] * yy.ravel() + b)
Z = (Z >= 0.5).astype(int)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    edgecolors="black",
    cmap="coolwarm"
)
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

thresholds = np.concatenate((
    [np.inf],
    np.sort(np.unique(y_prob))[::-1],
    [-np.inf]
))
fpr = []
tpr = []
for threshold in thresholds:
    pred = (y_prob >= threshold).astype(int)
    TP = np.sum((y_test == 1) & (pred == 1))
    FP = np.sum((y_test == 0) & (pred == 1))
    TN = np.sum((y_test == 0) & (pred == 0))
    FN = np.sum((y_test == 1) & (pred == 0))
    tpr.append(TP / (TP + FN))
    fpr.append(FP / (FP + TN))
fpr = np.array(fpr)
tpr = np.array(tpr)
roc_auc = np.sum(
    (fpr[1:] - fpr[:-1]) *
    (tpr[1:] + tpr[:-1]) / 2
)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label="ROC Curve (AUC = %.2f)" % roc_auc)
plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    label="Random Classifier"
)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()
print("AUC Score:", roc_auc)


"""
=================== OUTPUT (text output; plots open in a window when run normally) ===================
Libraries Imported Successfully
Dataset Created Successfully
X Shape: (200, 2)
Y Shape: (200,)
Training Data: (160, 2)
Testing Data: (40, 2)
Logistic Regression Model Trained Successfully
Accuracy: 100.0 %
AUC Score: 1.0
"""
