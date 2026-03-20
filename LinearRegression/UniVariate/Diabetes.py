import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from sklearn import linear_model      ## from sklearn import linear_model, datasets
from sklearn import datasets
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()
# print(diabetes.DESCR)       
# print(diabetes.keys())
# print(diabetes.data)          ## FEATURES
# print(diabetes.target)        ## LABEL

# print(diabetes.data.shape)      ## (442,10)

# First 1 Record
# print(diabetes.data[0])

# last 1 Record
# print(diabetes.data[441])

# First 3 record
# x= diabetes.data[:3]
# print(x,x.shape)

# diabetes_X=diabetes.data[:,np.newaxis]
# print(diabetes_X.shape) 

# diabetes_X=diabetes.data[:,np.newaxis,0]
# print(diabetes_X[:10]) 

diabetes_X=diabetes.data[:,np.newaxis,2]
diabetes_X_train=diabetes_X[:-30]
# will take all except last 30 values
# print(diabetes_X_train.shape)        ##(412, 1)
diabetes_X_test=diabetes_X[-30:]
# print(diabetes_X_test.shape)         ## (30, 1)


diabetes_y=diabetes.data[:,np.newaxis,9]
diabetes_y_train=diabetes_y[:-30]
# print(diabetes_y_train.shape)        ##(412, 1)
diabetes_y_test=diabetes_y[-30:]
# print(diabetes_y_test.shape)         ##(30, 1)

## Training Model
lr_model= linear_model.LinearRegression()
lr_model.fit(diabetes_X_train,diabetes_y_train)

## Prediction
diabetes_y_predicted=lr_model.predict(diabetes_X_test)
# print(diabetes_y_predicted)
# print(diabetes_y_predicted.shape)

## Plotting

plt.scatter(diabetes_X_train,diabetes_y_train)
plt.xlabel("Diabetes_X",fontsize=20)
plt.ylabel("Diabetes_y",fontsize=20)
plt.plot(diabetes_X_test,diabetes_y_predicted,color="red")
plt.plot(diabetes_X_test,diabetes_y_test,color="green")
plt.show()

print("New Squared Error Is : ",mean_squared_error(diabetes_y_test,diabetes_y_predicted))
