from dataset.iris_dataset import IrisDataset
from sklearn.linear_model import LogisticRegression
import numpy as np
from model.LogicReg import LogisticRegression
from Visualise.myVis import myplot 

IRIS_DIR_PATH = "Data"
queries = ["Species == 'Iris-setosa' | Species == 'Iris-versicolor'"]
features=["SepalLengthCm","SepalWidthCm"]
labels=["Species"]
  
def main():
    (X_train, y_train),(X_test, y_test)=IrisDataset.load_data(IRIS_DIR_PATH,queries,features,labels)
    # print("X_train : ",X_train.shape)           ## (75, 2)
    # print("y_train : ",y_train.shape)           ## (75, 1)
    # print("X_test : ",X_test.shape)             ## (25, 2)
    # print("y_test : ",y_test.shape)             ## (25, 1)
    # model = LogisticRegression()
    # model.fit(X_train, y_train)  
    # y_pred = model.predict(X_test)
    # print(y_pred)
    
    # print(y_train[:5])
    
    model = LogisticRegression(learning_rate=0.3,max_itr=500)
    losses= model.fit(X_train,y_train)
    # print(losses)
    y_pred_train=model.predict(X_train)
    # print(y_train[:5])
    # print(y_pred_train[:5])
    y_pred_test=model.predict(X_test)
    # print(y_pred_test[:5])
    # print(y_test[:5])
    # print("ACCURACY ON TRAINING DATA :",model.accuracy(y_train,y_pred_train))
    # print("ACCURACY ON TRAINING DATA :",model.accuracy(y_test,y_pred_test))
   
    ## VISUALIZE STARTS 
    # myplot.plot(losses,"max_itr","Losses")
    myplot.plot_labelpoints(X_train,np.reshape(y_train,newshape=(y_train.shape[0])))
    myplot.plot_featurepoints(X_train,np.reshape(y_train,newshape=(y_train.shape[0])))


if __name__ =='__main__':
    main()
