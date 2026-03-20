import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/diabetes.csv")
# print(df.shape)        ##(768, 9)
# print(df.columns)

# i=1
# for column in df:
#     print(i,".",column)
#     i+=1

X=df.drop("Outcome",axis=1)
# print(X.shape)               ##(768, 8)
y=df[['Outcome']]
# print(y.shape)               ##(768,1)

# print(X[:3])     ## first 3 

# X= X- axis / Xmax-Xmin
# scaler= MinMaxScaler()
# X= scaler.fit_transform(X)
# print(X[:3])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
# print("X_train --> ",X_train.shape," X_test.shape",X_test.shape," y_train",y_train.shape,"y_test",y_test.shape)

## Testing
lr_model= LinearRegression()
lr_model.fit(X_train,y_train)
pred_y_diabetes=lr_model.predict(X_test)
# print(pred_y_diabetes[:5])

pred=lr_model.predict([[6,148,72,35,0,33.6,0.627,50]])
# print(pred)

Preganancies = input("Enter Preganancies: ")
Preganancies = int(Preganancies)
Glucose = input("Enter Glucose: ")
Glucose = int(Glucose)
BloodPressure = input("Enter BloodPressure: ")
BloodPressure = int(BloodPressure)
SkinThickness = input("Enter SkinThickness: ")
SkinThickness = int(SkinThickness)
Insulin = input("Enter Insulin: ")
Insulin = int(Insulin)
BMI = input("Enter BMI: ")
BMI = float(BMI)
DiabetesPedigreeFunction = input("Enter DiabetesPedigreeFunction: ")
DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
Age = input("Enter Age: ")
Age = int(Age)
pred = lr_model.predict([[Preganancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
print("Prediction for the given input: ", pred)
