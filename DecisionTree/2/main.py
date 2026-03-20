import numpy as np
from  dataset.iris_dataset import IrisDataset
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz

IRIS_DIR_PATH = "Data"
queries = ["Species == 'Iris-setosa' | Species == 'Iris-versicolor' | Species == 'Iris-virginica'"]
features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
labels=["Species"]

def main():
    (X_train,y_train),(X_test,y_test)=IrisDataset.load_data(IRIS_DIR_PATH,queries,features,labels,scale=True)
    # print("X_train : ",X_train.shape)           ## (112, 4)
    # print("y_train : ",y_train.shape)           ## (112, 1)
    # print("X_test : ",X_test.shape)             ## (38, 4)
    # print("y_test : ",y_test.shape)             ## (38, 1)
    
    from sklearn import tree
    model=tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
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
            
    dot_data = tree.export_graphviz( model,out_file=None,feature_names=features)
    graph = graphviz.Source(dot_data)
    graph.render("iris.gv",view=True)
    
if __name__ =='__main__':
    main()

