# Import Libraries 

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import pickle


# Load Dataset 
emails = pd.read_csv('email.csv', encoding='latin-1')
# print(emails.shape)
# Data Cleaning 
emails = emails[['Category', 'Message']]

emails = emails.rename(columns={
    'Category': 'label',
    'Message': 'message'
})


# Convert Labels 

emails['label'] = emails['label'].map({
    'spam' : 1,
    'ham' : 0
})

# print(emails.isnull().sum())
emails = emails.dropna()
# print(emails.isnull().sum())
emails['label'] = emails['label'].astype(int)
# print(emails.head())


# Split Data 
X = emails["message"]
y = emails['label']

# print(Y)
# print(X)


# Text → Numbers (Vectorization)  
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)
# print(X)



# Train/Test Split  
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# print(X_train.shape)
# print(X_test.shape)


# Train Model 

model = MultinomialNB()
model.fit(X_train,y_train)

# Test Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)