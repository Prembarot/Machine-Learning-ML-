## Data science ~ pre processing
import pandas as pd
import numpy as np

df=pd.read_csv("data\homeprices.csv")

##print(df)

# print(df["area"])
# print(df["price"])

## DATA SCIENCE

## MACHINE LEARNING
from sklearn import linear_model

print("Home Prediction Model, for Single Area Prediction")
lr_model= linear_model.LinearRegression()
x=df[["area"]]
# print("Shape of Features : ",x.shape)
y=df["price"]
# print("Shape of Labels :",y.shape)
lr_model.fit(x,y)

# result = lr_model.predict([[3300]])
area=int(input("Enter Area in square feet : "))
result = lr_model.predict([[area]])
print("Prediction vale is ",result)
 
