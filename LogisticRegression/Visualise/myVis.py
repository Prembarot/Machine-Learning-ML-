import matplotlib.pyplot as plt
import numpy as mp
class myplot:
    def plot(data=[],x_label="",y_label=""):
        # print("Plotting Starts ")
        # print(data)
        # print(x_label)
        # print(y_label)
        plt.plot(data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        
    def plot_labelpoints(X_train,y_train):
        plt.scatter(X_train[:,0],y_train)
        plt.show()
        plt.scatter(X_train[:,1],y_train)
        plt.show()
    
    def plot_featurepoints(X_train,y_train,x=0,y=1):
        colors=["r","g"]
        for i in range(2):
            # print(i)
            # print(y_train==i)
            data=X_train[y_train==i]
            plt.scatter(data[:,x],data[:,y],c=colors[i],s=50)
        plt.show()
        