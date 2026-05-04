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
x = emails["message"]
y = emails['label']

# print(y)
# print(x)







