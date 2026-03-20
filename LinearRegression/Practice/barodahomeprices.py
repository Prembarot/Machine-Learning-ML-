import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math

df = pd.read_csv("data/barodahomeprices1.csv")
# print(df)
# print(df.shape)

dumies=pd.get_dummies(df.town)
# print(dumies)

df_dumies= pd.concat([df,dumies],axis=1)
# print(df_dumies)

df_dumies.drop(["town"],axis=1,inplace=True)
# print(df_dumies)

df_dumies.drop(["Gotri"],axis=1,inplace=True)
# print(df_dumies)

X = df_dumies[["area", "Bhayli", "Karelibaug"]]
y = df_dumies["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

homesize=int(input("Enter Area (sq ft): "))
areaname = input("Enter area name: \nB. Bhayli\nK. Karelibaug\nG. Gotri\n")

K=0
B=0
G=0

if areaname == "b" or areaname == "B":
    B = 1
    print("Bhayli")
elif areaname == "k" or areaname == "K":
    K = 1
    print("Karelibaug")
else:
    G= 1
    print("Gotri")

prediction = model.predict([[homesize, B, K]])
print("\nPredicted Price: $", int(prediction[0]))




