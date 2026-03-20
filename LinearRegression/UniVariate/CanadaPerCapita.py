## WAP of LR predicting per capita income (per person income in country) of year specified also plot graph 
## with year on x-axis and income in rupeesin y-axis

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from sklearn import linear_model

#  Data Pre-Processing
df=pd.read_csv("data/canada_per_capita_income.csv")
# print(df.head())
# print("Last 5 elemnts : ",df.tail())
# print("Shape : ",df.shape)


# Training 
lr_model= linear_model.LinearRegression()
x=df[["year"]]
y=df["per_capita_income"]
lr_model.fit(x,y)

# Prediction 
year=int(input("Enter Year : "))
result = lr_model.predict([[year]])
print(f"Predicted Per Capita Income for {year} is : ", result)

# Plotting 
plt.scatter(df.year,df.per_capita_income)
plt.xlabel("year",fontsize=20)
plt.ylabel("income rupees)",fontsize=20)
plt.title("Per capita income (per person income in country) ") 
plt.show()

