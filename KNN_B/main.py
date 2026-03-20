import numpy as np
from  dataset.iris_dataset import IrisDataset
from model.Classification.knn_classification import KNN_Classification
from sklearn.neighbors._regression import KNeighborsRegressor
from sklearn import linear_model

IRIS_DIR_PATH = "Data"
queries = ["Species == 'Iris-setosa' | Species == 'Iris-versicolor' | Species == 'Iris-virginica'"]
features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
labels=["Species"]

def main():
    (X_train,y_train),(X_test,y_test)=IrisDataset.load_data(IRIS_DIR_PATH,queries,features,labels,scale=True)
    model= KNN_Classification(n_neighbours=5)
    print("My KNN Model")
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    print("Accuracy on Testing Data : ",model.accuracy(y_test,y_pred))
    print("Accuracy on Training Data : ",model.accuracy(y_train,y_pred_train))   
    
    print("KNN MODEL")
    model= KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train,y_train)
    print("Train Score")
    y_pred_train=model.predict(X_train)
    print(model.score(X_train,y_pred_train))
    print("Test Score") 
    y_pred_train=model.predict(X_test)
    print(model.score(X_test,y_pred_test))
    
    print("LINEAR REG MODEL")
    model= linear_model.LinearRegression()
    model.fit(X_train,y_train)
    print("Train Score")
    y_pred_train=model.predict(X_train)
    print(model.score(X_train,y_pred_train))
    print("Test Score") 
    y_pred_train=model.predict(X_test)
    print(model.score(X_test,y_pred_test))
    
    
if __name__ =='__main__':
    main()
