import pandas as pd
import numpy as np


df = pd.read_csv("email.csv")


# replace the column names with more descriptive names 

df.rename(columns={'Category': 'Target'}, inplace=True)
print(df.head())