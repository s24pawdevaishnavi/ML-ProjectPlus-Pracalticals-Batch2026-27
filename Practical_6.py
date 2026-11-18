# Practical No. 6
# Title: Implement & visualize K-NN classification, tune k & compare accuracy, demonstrate curse of dimensionality

import numpy as np
import matplotlib.pyplot as plt
print("Libraries Imported Successfully")

np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 4) + [5, 3, 1, 1],
    np.random.randn(50, 4) + [6, 4, 4, 4],
    np.random.randn(50, 4) + [7, 2, 6, 5]
])
y = np.array([0]*50 + [1]*50 + [2]*50)
print("Dataset Loaded Successfully")
print("X Shape:", X.shape)
print("Y Shape:", y.shape)

np.random.seed(42)
train_indices = []
test_indices = []
for cls in np.unique(y):
    indices = np.where(y == cls)[0]
    np.random.shuffle(indices)
    test_indices.extend(indices[:10])
    train_indices.extend(indices[10:])
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
print("Data Scaled Successfully")

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        nearest = np.argsort(distances)[:k]
        labels = y_train[nearest]
        prediction = np.bincount(labels).argmax()
        predictions.append(prediction)
    return np.array(predictions)
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
accuracies = []
print("KNN CLASSIFICATION RESULTS")
print("--------------------------")
for k in k_values:
    y_pred = knn_predict(X_train, y_train, X_test, k)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print("K =", k, "Accuracy =", round(accuracy, 4))

best_index = np.argmax(accuracies)
best_k = k_values[best_index]
print("Best K Value:", best_k)
print("Best Accuracy:", round(accuracies[best_index], 4))

plt.figure(figsize=(7, 5))
plt.plot(
    k_values,
    accuracies,
    marker='o'
)
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different K Values")
plt.grid()
plt.show()

model_predictions = knn_predict(
    X_train[:, :2],
    y_train,
    X_test[:, :2],
    best_k
)
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = knn_predict(
    X_train[:, :2],
    y_train,
    grid,
    best_k
)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    edgecolors='black'
)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KNN Classification Visualization")
plt.show()

print("CURSE OF DIMENSIONALITY")
print("-----------------------")
dimensions = [2, 5, 10, 20, 40, 80]
dimension_accuracy = []
for d in dimensions:
    np.random.seed(42)
    n_samples = 500
    y_dim = np.array([0]*250 + [1]*250)
    X_dim = np.random.randn(n_samples, d)
    X_dim[:250, :2] -= 1
    X_dim[250:, :2] += 1
    indices = np.random.RandomState(42).permutation(n_samples)
    test_indices = indices[:100]
    train_indices = indices[100:]
    Xtr = X_dim[train_indices]
    Xte = X_dim[test_indices]
    ytr = y_dim[train_indices]
    yte = y_dim[test_indices]
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    pred = knn_predict(Xtr, ytr, Xte, 5)
    acc = accuracy_score(yte, pred)
    dimension_accuracy.append(acc)
    print("Dimensions:", d, "Accuracy:", round(acc, 4))

plt.figure(figsize=(7, 5))
plt.plot(
    dimensions,
    dimension_accuracy,
    marker='o'
)
plt.xlabel("Number of Dimensions")
plt.ylabel("Accuracy")
plt.title("Curse of Dimensionality")
plt.grid()
plt.show()


"""
=================== OUTPUT (text output; plots open in a window when run normally) ===================
Libraries Imported Successfully
Dataset Loaded Successfully
X Shape: (150, 4)
Y Shape: (150,)
Training Data: (120, 4)
Testing Data: (30, 4)
Data Scaled Successfully
KNN CLASSIFICATION RESULTS
--------------------------
K = 1 Accuracy = 0.9333
K = 3 Accuracy = 0.9333
K = 5 Accuracy = 0.9333
K = 7 Accuracy = 0.9333
K = 9 Accuracy = 0.8667
K = 11 Accuracy = 0.8667
K = 13 Accuracy = 0.9
K = 15 Accuracy = 0.9
Best K Value: 1
Best Accuracy: 0.9333
CURSE OF DIMENSIONALITY
-----------------------
Dimensions: 2 Accuracy: 0.9
Dimensions: 5 Accuracy: 0.94
Dimensions: 10 Accuracy: 0.88
Dimensions: 20 Accuracy: 0.83
Dimensions: 40 Accuracy: 0.81
Dimensions: 80 Accuracy: 0.72
"""
