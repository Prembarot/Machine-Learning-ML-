import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from word2number import w2n
import math

df = pd.read_csv("data/hiring.csv")
# print(df)
# print(df.shape)
df["experience"].fillna("Zero", inplace=True)
# print(df)
df.experience = df.experience.apply(w2n.word_to_num)
# print(df)
# df["test_score(out of 10)"].fillna("0", inplace=True)
# print(df)
# df['test_score(out of 10)'] = df['test_score(out of 10)'].replace(0, np.nan)
# print(df)
# df["test_score(out of 10)"].fillna(df["test_score(out of 10)"].mean(), inplace=True)
# print(df)

floor_mean = math.floor(df["test_score(out of 10)"].mean())
df["test_score(out of 10)"].fillna(floor_mean, inplace=True)
# print(df)

X = df[["experience", "test_score(out of 10)", "interview_score(out of 10)"]]
y = df["salary($)"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

exp = int(input("Enter Experience: "))
test = int(input("Enter Test Score (out of 10): "))
interview = int(input("Enter Interview Score (out of 10): "))

prediction = model.predict([[exp, test, interview]])

print("\nPredicted Salary: $", int(prediction[0]))