from sklearn import linear_model

# regression model
reg = linear_model.LinearRegression()


## data
## x ==> Represt Features ==> 2D array
## Y ==> Represt Labels ==> 1D array

x = [[1],[2],[3],[4],[5],[6]]  ## features
y = [2,2.5,4.5,3,5,4.7]        ## labels

## Training Model

reg.fit(x,y)

## Prediction 

result = reg.predict([[5.5]])
print("Prediction vale is ",result)



