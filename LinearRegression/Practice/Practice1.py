import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/carprices.csv")
# print(df[:5])
plt.xlabel("Mileage")
plt.ylabel(" Sell Price")
# plt.scatter(df["Mileage"], df["Sell Price($)"])
# plt.show()

# plt.scatter(df["Age(yrs)"], df["Sell Price($)"])
# plt.show()

# X=df[["Mileage","Age(yrs)"]]
# y=df["Sell Price($)"]
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
# lr_model= LinearRegression()
# lr_model.fit(X_train,y_train)
# pred=lr_model.predict(X_test)
# print(pred[:5])

X=df[["Mileage","Age(yrs)"]]
y=df[["Sell Price($)"]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)
lr_model= LinearRegression()
lr_model.fit(X_train,y_train)
pred=lr_model.predict(X_test)
print(pred)
print(pred.shape)

