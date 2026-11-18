# Practical No. 4
# Title: Train a decision tree classifier and visualize the tree. Analyze performance with and without depth control against overfitting

from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree  # NOTE: not visible in the scanned page (first cell cut off); added because the code needs it
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
print("Test Accuracy: ", accuracy_score(y_test, tree.predict(X_test)))

plt.figure(figsize=(12, 7))

plot_tree(
    tree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True)
plt.show()

for d in [2, 3, 5, None]:
    model = DecisionTreeClassifier(
        max_depth=d,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Depth: ", d,
          "Train: ", round(model.score(X_train, y_train), 2),
          "Test: ", round(model.score(X_train, y_train), 2)
          )

for alpha in [0, 0.01, 0.02]:
    model = DecisionTreeClassifier(
        ccp_alpha=alpha,
        random_state=42)
    model.fit(X_train, y_train)
    print(
        "Pruning:", alpha,
        "Train:", round(model.score(X_train, y_train), 2),
        "Test: ", round(model.score(X_train, y_train), 2)
    )

X, y = make_moons(
    n_samples=300,
    noise=0.3,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

deep_tree = DecisionTreeClassifier(random_state=42)
deep_tree.fit(X_train, y_train)
print("Training Accuracy: ", deep_tree.score(X_train, y_train))
print("Testing Accuracy: ", deep_tree.score(X_test, y_test))

small_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
small_tree.fit(X_train, y_train)
print("Training Accuracy: ", small_tree.score(X_train, y_train))
print("Testing Accuracy:", small_tree.score(X_test, y_test))


"""
=================== OUTPUT (text output; plots open in a window when run normally) ===================
Test Accuracy:  1.0
Depth:  2 Train:  0.94 Test:  0.94
Depth:  3 Train:  0.95 Test:  0.95
Depth:  5 Train:  0.99 Test:  0.99
Depth:  None Train:  1.0 Test:  1.0
Pruning: 0 Train: 1.0 Test:  1.0
Pruning: 0.01 Train: 0.99 Test:  0.99
Pruning: 0.02 Train: 0.94 Test:  0.94
Training Accuracy:  1.0
Testing Accuracy:  0.8555555555555555
Training Accuracy:  0.8952380952380953
Testing Accuracy: 0.9
"""
