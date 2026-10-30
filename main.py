import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("email.csv")

# print("FIRST 5 ROWS OF THE DATASET")
# print(df.head())

# print("LAST 5 ROWS OF THE DATASET")
# print(df.tail())

# print("SHAPE OF THE DATASET")
# print(df.shape)

# print("INFO OF THE DATASET")
# print(df.info())

# print("DESCRIPTION OF THE DATASET")
# print(df.describe())


# print("VALUE COUNTS OF THE FIRST COLUMN")
# print(df.iloc[:, 0].value_counts())

# print("VALUE COUNTS OF THE SECOND COLUMN")
# print(df.iloc[:, 1].value_counts())

le = LabelEncoder()

df["Category"] = le.fit_transform(df["Category"])

duplicates = df[df.duplicated()]

# print(duplicates.head(10))
# print("Total duplicates:", len(duplicates))


duplicate_messages = df[
    df["Message"].duplicated(keep=False)
].sort_values("Message")

# print(duplicate_messages.head(20))


df = df.drop_duplicates()

print("Duplicates remaining:", df.duplicated().sum())
print("New dataset shape:", df.shape)




print("Ham vs Spam Distribution")
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Category")

plt.title("Ham vs Spam Distribution")
plt.xlabel("Category (0 = Ham, 1 = Spam)")
plt.ylabel("Number of Emails")

plt.show()


print(df["Category"].value_counts(normalize=True) * 100)