# Customer Review Sentiment Analysis

## Project objective
Classify an English customer review as **positive** or **negative** using supervised machine learning.

## Folder contents
- `reviews.csv` — sample labeled dataset (60 reviews)
- `train_model.py` — preprocessing, training, evaluation and model saving
- `app.py` — Streamlit web interface
- `requirements.txt` — required Python packages
- `PROJECT_REPORT.md` — editable mini-project report
- `VIVA_QUESTIONS.md` — viva preparation

## Requirements
Python 3.9 or later and an internet connection for the first-time package installation.

## Run in VS Code / terminal
1. Extract this ZIP file.
2. Open the extracted folder in VS Code.
3. Create and activate a virtual environment (optional but recommended):
   - Windows: `py -m venv venv`
   - Windows activation: `venv\\Scripts\\activate`
4. Install packages: `pip install -r requirements.txt`
5. Train and evaluate: `python train_model.py`
6. Start the website: `streamlit run app.py`
7. Open the local URL printed in the terminal (usually `http://localhost:8501`).

Run `train_model.py` before starting the website. It creates `sentiment_model.joblib`, `evaluation_results.json`, and `confusion_matrix.png`.

## Method
- Load labeled review text from CSV.
- Split into training and testing sets (75% / 25%).
- Convert text into TF-IDF numerical features.
- Train Multinomial Naive Bayes and Logistic Regression pipelines.
- Compare test accuracy and classification reports.
- Save the better-performing model and use it in the Streamlit app.

## Important academic note
The included 60-row dataset is a small demonstration dataset created for learning. It is not representative of all real customer opinions. The actual accuracy must be taken from the output produced on your computer; do not claim a guessed accuracy. For a stronger submission, replace or expand `reviews.csv` with a larger, properly labeled dataset and repeat the evaluation.
