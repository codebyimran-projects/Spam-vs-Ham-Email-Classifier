import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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