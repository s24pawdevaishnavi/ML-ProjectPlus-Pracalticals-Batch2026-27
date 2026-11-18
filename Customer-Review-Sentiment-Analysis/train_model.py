"""
Train and evaluate two sentiment-classification models.
Run: python train_model.py
"""
import json
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_FILE = "reviews.csv"
MODEL_FILE = "sentiment_model.joblib"

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("reviews.csv not found. Keep it in the same folder as this script.")

    data = pd.read_csv(DATA_FILE).dropna()
    data["sentiment"] = data["sentiment"].str.lower().str.strip()
    X = data["review"].astype(str)
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("classifier", MultinomialNB())
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])
    }

    results = {}
    best_name, best_model, best_accuracy = None, None, -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = {
            "accuracy": round(float(accuracy), 4),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )
        }
        print(f"\n{name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(classification_report(y_test, predictions, zero_division=0))

        if accuracy > best_accuracy:
            best_name, best_model, best_accuracy = name, model, accuracy
            best_predictions = predictions

    joblib.dump(best_model, MODEL_FILE)
    print(f"Best model saved: {MODEL_FILE} ({best_name})")

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "models": results}, f, indent=2)

    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, best_predictions, labels=["negative", "positive"]),
        display_labels=["Negative", "Positive"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved evaluation_results.json and confusion_matrix.png")

if __name__ == "__main__":
    main()
