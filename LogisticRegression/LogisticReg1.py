import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data\\insurance_data.csv")
# print(df)

# plt.scatter(df.age,df.bought_insurance,marker="+",color='g')
# plt.show()

X = df[['age']]   
# print(X.shape)                       ## (27, 1)        
y = df[['bought_insurance']]         ##  (27, 1)
# print(y.shape)    

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)

# print("X_train :", X_train.shape)        ## (24, 1)
# print("X_test :", X_test.shape)          ## (3, 1)
# print("y_train :", y_train.shape)        ## (24, 1)
# print("y_test :", y_test.shape)          ## (3, 1)

model = LogisticRegression()
model.fit(X_train, y_train)
age = int(input("Enter age: "))
y_pred = model.predict([[age]])
# print("Predicted:", y_pred)
# print(model.predict_proba([[age]]))

print(X_test)
print(model.predict(X_test))
print(model.predict_proba(X_test))

