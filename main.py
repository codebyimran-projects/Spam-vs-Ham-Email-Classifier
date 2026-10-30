
# ============================================================
# PROJECT: EMAIL SPAM CLASSIFIER
# STAGE 1: DATASET LOADING, UNDERSTANDING & CLEANING
# ============================================================


# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# 2. LOAD THE DATASET
# ------------------------------------------------------------

df = pd.read_csv("email.csv")

print("\n========== DATASET LOADED SUCCESSFULLY ==========")


# 3. UNDERSTAND THE DATASET
# ------------------------------------------------------------

# Display the first five records
print("\nFIRST 5 ROWS:")
print(df.head())

# Display the last five records
print("\nLAST 5 ROWS:")
print(df.tail())

# Display the number of rows and columns
print("\nDATASET SHAPE:")
print(df.shape)

# Display column names, data types and non-null values
print("\nDATASET INFORMATION:")
df.info()

# Display statistical information
print("\nDATASET DESCRIPTION:")
print(df.describe(include="all"))

# Display the distribution of original category labels
print("\nORIGINAL CATEGORY DISTRIBUTION:")
print(df["Category"].value_counts())


# 4. ENCODE CATEGORY LABELS
# ------------------------------------------------------------

# Convert text labels into numerical values.
# LabelEncoder assigns numbers based on alphabetical order.
# For the labels 'ham' and 'spam':
# ham  = 0
# spam = 1

le = LabelEncoder()

df["Category"] = le.fit_transform(df["Category"])

print("\nENCODED CATEGORY LABELS:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

print("\nCATEGORY DISTRIBUTION AFTER ENCODING:")
print(df["Category"].value_counts())


# 5. IDENTIFY DUPLICATE RECORDS
# ------------------------------------------------------------

# Find rows where all column values are duplicated
duplicates = df[df.duplicated()]

print("\nDUPLICATE RECORDS:")
print(duplicates.head(10))

print("\nTOTAL DUPLICATE RECORDS:")
print(len(duplicates))


# 6. IDENTIFY DUPLICATE EMAIL MESSAGES
# ------------------------------------------------------------

# Find repeated messages, even if their categories differ.
# keep=False marks every occurrence of a repeated message.

duplicate_messages = df[
    df["Message"].duplicated(keep=False)
].sort_values("Message")

print("\nDUPLICATE EMAIL MESSAGES:")
print(duplicate_messages.head(20))

print("\nTOTAL ROWS WITH REPEATED MESSAGES:")
print(len(duplicate_messages))


# 7. REMOVE DUPLICATE RECORDS
# ------------------------------------------------------------

# Remove rows that are completely identical.
# Keep the first occurrence of each duplicate record.

df = df.drop_duplicates()

# Reset the index after removing duplicates
df = df.reset_index(drop=True)

print("\nDATASET AFTER DUPLICATE REMOVAL:")
print("Remaining duplicates:", df.duplicated().sum())
print("Updated dataset shape:", df.shape)


# 8. REMOVE UNKNOWN CATEGORY VALUES
# ------------------------------------------------------------

# This is a binary classification project:
# 0 = Ham
# 1 = Spam
#
# Keep only valid categories.
# Remove records with unexpected category values.

df = df[df["Category"].isin([0, 1])].copy()

# Reset the index after filtering
df = df.reset_index(drop=True)

print("\nDATASET AFTER CATEGORY CLEANING:")

print("\nCATEGORY COUNTS:")
print(df["Category"].value_counts().sort_index())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nUNIQUE CATEGORY VALUES:")
print(df["Category"].unique())


# 9. VISUALIZE HAM VS SPAM DISTRIBUTION
# ------------------------------------------------------------

# Count the number of emails in each category.
# sort_index() ensures 0 comes before 1.

category_counts = df["Category"].value_counts().sort_index()

# Create the pie chart
plt.figure(figsize=(7, 7))

plt.pie(
    category_counts,
    labels=["Ham", "Spam"],
    autopct="%1.1f%%",
    startangle=90,
    explode=(0, 0.05),
    shadow=True,
    textprops={"fontsize": 12}
)

plt.title(
    "Ham vs Spam Email Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.axis("equal")
plt.show()


# 10. FINAL DATASET SUMMARY
# ------------------------------------------------------------

print("\n========== FINAL DATASET SUMMARY ==========")

print("Total Emails:", len(df))
print("Total Columns:", len(df.columns))
print("Ham Emails:", (df["Category"] == 0).sum())
print("Spam Emails:", (df["Category"] == 1).sum())
print("Remaining Duplicates:", df.duplicated().sum())
print("Final Categories:", df["Category"].unique())

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")




nltk.download("punkt")
nltk.download("punkt_tab")


# ============================================================
# TEXT ANALYSIS USING NLTK
# ============================================================

# 1. WORD COUNT
# Count the total number of words in each email

df["Word_Count"] = df["Message"].apply(
    lambda message: len(word_tokenize(message))
)


# 2. SENTENCE COUNT
# Count the total number of sentences in each email

df["Sentence_Count"] = df["Message"].apply(
    lambda message: len(sent_tokenize(message))
)


# 3. CHARACTER COUNT
# Count the total number of characters in each email

df["Character_Count"] = df["Message"].apply(
    lambda message: len(message)
)


# 4. ALPHABET COUNT
# Count only alphabetic characters (A-Z and a-z)

df["Alphabet_Count"] = df["Message"].apply(
    lambda message: sum(char.isalpha() for char in message)
)


# DISPLAY RESULTS

print("\n========== TEXT ANALYSIS RESULTS ==========")

print(
    df[
        [
            "Category",
            "Message",
            "Word_Count",
            "Sentence_Count",
            "Character_Count",
            "Alphabet_Count"
        ]
    ].head(10)
)
