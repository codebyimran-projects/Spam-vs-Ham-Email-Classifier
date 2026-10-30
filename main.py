
# ============================================================
# EMAIL SPAM CLASSIFIER
# FEATURE EXTRACTION & MODEL TRAINING
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    classification_report,
    confusion_matrix
)


# 1. LOAD AND CLEAN DATASET

df = pd.read_csv("email.csv")

df = df.dropna(subset=["Category", "Message"]).copy()

df["Category"] = (
    df["Category"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert category labels into binary numerical values
df["Category"] = df["Category"].map({
    "ham": 0,
    "spam": 1,
    "0": 0,
    "1": 1
})

# Remove unknown categories and empty messages
df = df.dropna(subset=["Category"])

df["Message"] = (
    df["Message"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df = df[df["Message"] != ""]

# Remove duplicate records
df = df.drop_duplicates().reset_index(drop=True)

df["Category"] = df["Category"].astype(int)


# 2. SEPARATE FEATURES AND TARGET

X = df["Message"]
y = df["Category"]


# 3. SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. TEXT FEATURE EXTRACTION USING TF-IDF

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


# 5. INITIALIZE CLASSIFICATION MODELS

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


# 6. TRAIN AND EVALUATE MODELS

results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label=1, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model


# 7. COMPARE MODEL PERFORMANCE

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print("\nMODEL PERFORMANCE COMPARISON")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)


# 8. SELECT BEST MODEL

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBEST MODEL:", best_model_name)


# 9. CLASSIFICATION REPORT

best_predictions = best_model.predict(X_test_tfidf)

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)


# 10. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.tight_layout()
plt.show()


# 11. MODEL PERFORMANCE GRAPH

plot_data = results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
]

plot_data.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1),
    width=0.8
)

plt.title("Classification Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Machine Learning Models")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
