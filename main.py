
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# 1. LOAD DATASET

df = pd.read_csv("email.csv")

df = df.dropna(subset=["Category", "Message"]).copy()

df["Category"] = (
    df["Category"].astype(str).str.strip().str.lower()
)

df["Category"] = df["Category"].map({
    "ham": 0,
    "spam": 1,
    "0": 0,
    "1": 1
})

df = df.dropna(subset=["Category"])

df["Message"] = (
    df["Message"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df = df[df["Message"] != ""]
df = df.drop_duplicates().reset_index(drop=True)
df["Category"] = df["Category"].astype(int)


# 2. SPLIT FEATURES AND TARGET

X = df["Message"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 3. TF-IDF FEATURE EXTRACTION

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# 4. INITIALIZE MODELS

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}


# 5. TRAIN AND EVALUATE MODELS

results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

    trained_models[name] = model


# 6. SELECT BEST MODEL

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nMODEL PERFORMANCE:")
print(results_df.to_string(index=False))

print("\nBEST MODEL:", best_model_name)

best_predictions = best_model.predict(X_test_tfidf)

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)


# 7. SAVE MODEL AND VECTORIZER

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/spam_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

# Save selected model name and evaluation results
joblib.dump(
    {
        "model_name": best_model_name,
        "results": results_df.to_dict(orient="records")
    },
    "models/model_info.pkl"
)

print("\nMODEL SAVED SUCCESSFULLY!")
print("Saved model:", "models/spam_model.pkl")
print("Saved vectorizer:", "models/tfidf_vectorizer.pkl")
