import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self,learning_rate,max_itr):
        # print("Hi Its me learning ML from Amit Patni")
        self.learning_rate =learning_rate
        self.max_itr =max_itr
        self.theta= None
        # print("Learning rate:", self.learning_rate)
        # print("Iterations:", self.max_itr)
    
    def fit(self, X, y):
        losses=[]
        # print("Shape of X_train:", X.shape)
        # print("Shape of y_train:", y.shape)
        self.initialize_theta(X)
        for i in range(self.max_itr):
            d_theta=self.calc_grad(X,y)
            self.theta=self.theta-self.learning_rate*d_theta
            loss=self.loss(X,y)
            losses.append(loss)
        return losses
        
    def loss(self,X,y):
        y_pred=self.predict_probablity(X)
        # L = -[Y * log (Y) + (1-Y) log(1-Y)]
        return -np.average(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
    
    def initialize_theta(self,X):
        # print(X.shape)                 ## (75, 2)
        # print(X.shape[1])              ## 2
        n_features=X.shape[1]+1
        # print(n_features)              ## 3
        self.theta = np.zeros((n_features,1))
        # print(self.theta.shape)
    
    def calc_grad(self,X,y):
        # print(X.shape)                   ## (75, 2)
        # print(X[:5])
        y_pred=self.predict_probablity(X)
        tempX= self.add_ones(X)
        d_theta=np.average((y_pred-y)*tempX,axis=0)
        d_theta= d_theta.reshape(d_theta.shape[0],1)
        # print("d_theta shape : ",d_theta.shape)           ## (3, 1)
        # print("d_theta :",d_theta)                        ## [[ 0.09728395] [-0.055     ][ 0.02      ]]
        return d_theta             

    def predict_probablity(self,X):
        tempX= self.add_ones(X)
        z=np.matmul(tempX,self.theta)
        # print(z)
        # print("Z shape :",z.shape)
        return self.sigmoid(z)
    
    def sigmoid(self,z):
        return np.exp(z)/(1+np.exp(z))
        
    def add_ones(self,X):
        tempX = np.ones((X.shape[0], X.shape[1] + 1))
        # print("TempX shape :",tempX.shape)
        # print(tempX)
        tempX[:, :-1] = X
        tempX[:,0:X.shape[1]]=X
        # print(tempX)                       ## [[0.51851852 1.         1.        ]]
        return tempX
    
    def predict(self,X,threshold=0.5):
        y_pred=self.predict_probablity(X)
        y_hat=[]
        for y in y_pred:
            if y >= threshold:
                y_hat.append(1)
            else:
                y_hat.append(0)
        return np.array(y_hat)
    ## here v r converting list into numpy array
    
    def accuracy(slef,y_true,y_pred):
        return accuracy_score(y_true,y_pred)
        
        