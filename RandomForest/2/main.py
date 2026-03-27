import numpy as np
from  dataset.iris_dataset import IrisDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn 

IRIS_DIR_PATH = "Data"
queries = ["Species == 'Iris-setosa' | Species == 'Iris-versicolor' | Species == 'Iris-virginica'"]
features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
labels=["Species"]

def main():
    (X_train,y_train),(X_test,y_test)=IrisDataset.load_data(IRIS_DIR_PATH,queries,features,labels,scale=True)
    model= RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train)
    print(model.score(X_test,y_test))                ## 1.0
    y_pred_train= model.predict(X_train)
    # print(y_pred_train)
    y_pred_test= model.predict(X_test)
    # print(y_pred_test)
    # print("DECISION TREE SCORE")
    # print("Training Score :",model.score(X_train,y_pred_train))
    # print("Testing Score :",model.score(X_test,y_pred_test))
    # print("DECISION TREE ACCURACY SCORE")
    # print("Training accuracy :",accuracy_score(y_train,y_pred_train))
    # print("Testing accuracy :",accuracy_score(y_test,y_pred_test))
    
    print("Model score :",model.score(X_test,y_test))                ##
    print("ACTUAL VALUE :")
    print(y_test)
    print("PRICRED VALUE :")
    y_pred=model.predict(X_test)
    print(y_pred)
    COMMAT=confusion_matrix(y_test,y_pred)
    print("CONFUSION MATRIX :",COMMAT)
    plt.figure(figsize=(10,7))
    sn.heatmap(COMMAT,annot=True,cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


if __name__ =='__main__':
    main()
    
