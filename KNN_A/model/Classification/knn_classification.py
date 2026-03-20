import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

class KNN_Classification:
    def __init__(self,n_neighbours):
        # print(n_neighbours)
        self.n_neighbours=n_neighbours
        # print("Knn : ",self.n_neighbours)
        self.X=None
        self.y=None
        
    def fit(self, X, y):
        # print("shape of X_train",X.shape)           ## (112, 4)
        # print("shape of y_train",y.shape)           ## (112, 1)
        self.X=X
        self.y=y
        
    # def predict(self,X_test):
    #     y_pred=[]
    #     # print(X_test.shape)                           ## (38, 4)
    #     for test_X in X_test[:1]:                         ## iterate 38 times
    #         dist_array=[]
    #         for train_X in self.X:                    ## iterate 112 times
    #             dist=np.sum(np.square(test_X-train_X))
    #             dist_array.append(dist)
    #         # print("Length :",len(dist_array))            ## Length : 112
    #         # list.sort(dist_array)
    #         # print(dist_array[:self.n_neighbours])
    #         # print(self.y.values.shape)                     ## (112, 1)
    #         temp_y=self.y.values.reshape(self.y.values.shape[0])
    #         # print(temp_y.shape)                            ## (112,)
    #         d,y = (list(t) for t in  zip(*sorted(zip(dist_array,temp_y))))
    #         # print(d[:self.n_neighbours])
    #         # print(y[:self.n_neighbours])
    #         y_labels=y[:self.n_neighbours]
    #         # print("ylabels :",y_labels)
    #         b= Counter(y_labels)
    #         # print(b)
    #         class_name=b.most_common(1)[0][0]
    #         # print(class_name)
    #         y_pred.append(class_name)
    #     return y_pred
          
    def predict(self, X_test):
        y_pred = []
        for test_X in X_test:   # remove [:1]
            dist_array = []
            for train_X in self.X:
                dist = np.sum(np.square(test_X - train_X))
                dist_array.append(dist)
            temp_y = self.y.values.reshape(self.y.values.shape[0])
            d, y = (list(t) for t in zip(*sorted(zip(dist_array, temp_y))))
            y_labels = y[:self.n_neighbours]
            b = Counter(y_labels)
            class_name = b.most_common(1)[0][0]
            y_pred.append(class_name)
        return y_pred  
    
    def accuracy(self,y_true,y_pred):
        return accuracy_score(y_true,y_pred)
    