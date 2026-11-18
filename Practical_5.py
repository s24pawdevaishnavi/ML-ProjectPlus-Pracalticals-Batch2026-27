# Practical No. 5
# Title: Implement a toy classification problem using decision boundaries. Apply train/test split / K-Fold cross-validation to dataset (e.g. compare accuracy, F1 score)

import numpy as np
import matplotlib.pyplot as plt

print("Libraries Imported Successfully")

np.random.seed(42)
X0 = np.random.randn(50, 2) + [-2, -2]
X1 = np.random.randn(50, 2) + [2, 2]
X = np.vstack((X0, X1))
y = np.array([0]*50 + [1]*50)
print("Dataset Created Successfully")
print("Total Samples:", len(X))
print("Features:", X.shape[1])
print("Classes:", np.unique(y))

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
plt.title("Toy Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()
plt.show()

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
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
print("Data Standardized Successfully")

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
def train_model(X, y):
    w = np.zeros(X.shape[1])
    b = 0
    lr = 0.1
    epochs = 1000
    for epoch in range(epochs):
        z = np.dot(X, w) + b
        y_prob = sigmoid(z)
        error = y_prob - y
        dw = np.dot(X.T, error) / len(y)
        db = np.mean(error)
        w -= lr * dw
        b -= lr * db
    return w, b
w, b = train_model(X_train, y_train)
print("Logistic Regression Model Trained Successfully")

y_prob = sigmoid(np.dot(X_test, w) + b)
y_pred = (y_prob >= 0.5).astype(int)
print("Actual Output:", y_test)
print("Predicted Output:", y_pred)

TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = (2 * precision * recall / (precision + recall)
      if (precision + recall) > 0 else 0)
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = sigmoid(np.dot(grid, w) + b)
Z = (Z >= 0.5).astype(int)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap='coolwarm',
    edgecolors='black'
)
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0)
    return accuracy, precision, recall, f1
X_cv = X[train_indices]
y_cv = y[train_indices]
np.random.seed(42)
folds = []
for cls in np.unique(y_cv):
    indices = np.where(y_cv == cls)[0]
    np.random.shuffle(indices)
    folds.append(np.array_split(indices, 5))
cv_results = []
for fold in range(5):
    test_idx = np.concatenate([folds[c][fold] for c in range(2)])
    train_idx = np.setdiff1d(
        np.arange(len(X_cv)),
        test_idx
    )
    Xtr = X_cv[train_idx]
    Xte = X_cv[test_idx]
    ytr = y_cv[train_idx]
    yte = y_cv[test_idx]
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    w_fold, b_fold = train_model(Xtr, ytr)
    pred = (
        sigmoid(np.dot(Xte, w_fold) + b_fold) >= 0.5
    ).astype(int)
    metrics = calculate_metrics(yte, pred)
    cv_results.append(metrics)
    print("Fold", fold + 1)
    print("Accuracy:", round(metrics[0], 4))
    print("Precision:", round(metrics[1], 4))
    print("Recall:", round(metrics[2], 4))
    print("F1 Score:", round(metrics[3], 4))
    print("----------------------")

cv_results = np.array(cv_results)
metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]
average_scores = np.mean(cv_results, axis=0)
print("AVERAGE 5-FOLD CROSS-VALIDATION RESULTS")
print("---------------------------------------")
for name, score in zip(metric_names, average_scores):
    print(name, ":", round(score, 4))

plt.figure(figsize=(7, 5))
plt.bar(metric_names, average_scores)
plt.title("Comparison of Evaluation Metrics")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.grid(axis='y')
plt.show()


"""
=================== OUTPUT (text output; plots open in a window when run normally) ===================
Libraries Imported Successfully
Dataset Created Successfully
Total Samples: 100
Features: 2
Classes: [0 1]
Training Samples: 80
Testing Samples: 20
Data Standardized Successfully
Logistic Regression Model Trained Successfully
Actual Output: [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
Predicted Output: [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0
Fold 1
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0
----------------------
Fold 2
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0
----------------------
Fold 3
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0
----------------------
Fold 4
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0
----------------------
Fold 5
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0
----------------------
AVERAGE 5-FOLD CROSS-VALIDATION RESULTS
---------------------------------------
Accuracy : 1.0
Precision : 1.0
Recall : 1.0
F1 Score : 1.0
"""
