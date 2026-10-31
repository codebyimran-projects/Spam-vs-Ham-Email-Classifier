# Spam vs Ham Email Classifier

A Machine Learning-based Email Spam Classifier built with **Python, Scikit-learn, NLP, TF-IDF, and Streamlit**. This project predicts whether an email message is **Spam** or **Ham (Legitimate Email)** using trained machine learning models.

The application provides an interactive web interface where users can enter email messages and instantly receive classification results.

## Project Overview

Email spam is a common problem that affects communication and online security. This project uses Natural Language Processing (NLP) and supervised Machine Learning algorithms to identify unwanted email messages.

The system processes email text, converts it into numerical features using TF-IDF, and uses a trained classification model to predict whether a message is spam or legitimate.

## Features

- Spam and Ham email classification
- Real-time email prediction
- Natural Language Processing (NLP)
- TF-IDF text feature extraction
- Comparison of multiple Machine Learning models
- Automatic selection of the best-performing model
- Spam precision, recall, and F1-score evaluation
- Confusion matrix visualization
- Interactive Streamlit dashboard
- Prediction history
- Model information and performance metrics
- Dark-themed user interface

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| TF-IDF | Text feature extraction |
| NLTK | Text analysis and preprocessing |
| Streamlit | Web application |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Saving and loading trained models |

## Machine Learning Models

This project compares three classification algorithms:

**1. Multinomial Naive Bayes**

A probabilistic classification algorithm commonly used for text classification.

**2. Logistic Regression**

A supervised learning algorithm used to classify email messages into two categories.

**3. Linear Support Vector Classifier (LinearSVC)**

A linear classification algorithm that finds an optimal decision boundary between Spam and Ham messages.

### Model Selection

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The best-performing model is selected using the **Spam class F1-score**, helping balance precision and recall for spam detection.

## Dataset

The dataset used in this project is stored in:

`email.csv`

It contains two main columns:

| Column | Description |
|---|---|
| Category | Email classification label |
| Message | Actual email text |

### Classification Labels

| Label | Meaning |
|---|---|
| 0 | Ham (Legitimate Email) |
| 1 | Spam |

The dataset is cleaned before training by handling missing values, removing duplicate messages, and excluding invalid labels.

## Project Structure

```text
Spam-vs-Ham-Email-Classifier/
│
├── email.csv
│
├── main.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── models/
    ├── spam_model.pkl
    ├── tfidf_vectorizer.pkl
    └── model_info.pkl
```

## Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/codebyimran-projects/Spam-vs-Ham-Email-Classifier.git
```

### 2. Navigate to the Project Directory

```bash
cd Spam-vs-Ham-Email-Classifier
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the Machine Learning Models

```bash
python main.py
```

This script cleans the dataset, trains multiple classification models, evaluates their performance, selects the best model, and saves the trained model files inside the `models/` directory.

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## How the Project Works

The classification process follows these steps:

1. **Data Collection:** Load the email dataset from the CSV file.
2. **Data Cleaning:** Handle missing values, duplicates, and invalid labels.
3. **Text Processing:** Prepare email messages for machine learning.
4. **Feature Extraction:** Convert text into numerical features using TF-IDF.
5. **Model Training:** Train multiple machine learning classification algorithms.
6. **Model Evaluation:** Compare model performance using classification metrics.
7. **Model Selection:** Select the best model based on Spam F1-score.
8. **Model Saving:** Save the selected model and TF-IDF vectorizer using Joblib.
9. **Email Prediction:** Use the Streamlit application to classify new email messages.

## Application Usage

1. Start the Streamlit application.
2. Open the Email Classifier page.
3. Enter an email message in the text area.
4. Click the Analyze Email button.
5. View the prediction: Spam or Ham.
6. Visit Prediction History to review previous predictions.
7. Open Model Information to view the selected model and evaluation results.

## Example Predictions

**Ham Email:**

```text
Hello, your university assignment is due on Friday.
Please submit your completed work through the student portal.
```

Expected classification: Ham

**Spam Email:**

```text
CONGRATULATIONS! You have won $1,000,000.
Claim your FREE prize now before this limited offer expires!
```

Expected classification: Spam

*These are illustrative examples; actual predictions depend on the trained model.*

## Future Improvements

- Improve classification performance with larger datasets.
- Add email file upload functionality.
- Add persistent prediction history using a database.
- Add model probability calibration.
- Deploy the application online.
- Improve detection of phishing and fraudulent emails.

## Author

**Muhammad Imran**

GitHub: [@codebyimran-projects](https://github.com/codebyimran-projects)

Project Repository: [Spam vs Ham Email Classifier](https://github.com/codebyimran-projects/Spam-vs-Ham-Email-Classifier)

---

**Built with Python and Machine Learning.**
