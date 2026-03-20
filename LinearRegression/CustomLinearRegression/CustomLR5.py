import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize

data = np.genfromtxt("data/home.txt", delimiter=',')
# print(data.shape)             ## (47, 3)

# X= data[:,[0,1]]      ## Features
# print(X)
# print(X.shape)          ## (47, 2)

# y=data[:,[2]]           ## Label
# print(y)
# print(y.shape)          ## (47, 1)

# X1= data[:,[0]]
# X2= data[:,[1]]
# print(X1)
# print(X2)

# min_max_scaler=preprocessing.MinMaxScaler()
# scaled_X=min_max_scaler.fit_transform(X)
# print(scaled_X.shape)                 ##(47, 2)
# scaled_y=min_max_scaler.fit_transform(y)
# print(scaled_y.shape)                 ##(47, 1)
  
data= normalize(data,axis=0)
# print(data[:5])
# print(data.shape)                       ##(47, 3)

## HYPER PARAMETER

## BatcH Gradient_descent
learning_rate=0.09
max_itr=500

## Stochastic Gradient_Decent
s_learning_rate=0.06
s_max_itr=500

## Mini Batch  Gradient_Decent
mb_learning_rate=0.09
mb_max_itr=500
batch_size=16

X= data[:,[0,1]]
y=data[:,[2]]  

# print(data.shape[0])           ## 47
# print(data.shape[1])           ## 3
# temp=(np.zeros(data.shape[1])) 
# print(temp)

# temp_X=(np.ones(data.shape[1])) 
# print(temp_X)                   ## [1. 1. 1.]
# print(temp_X.shape)             ## (3,)

temp_X=np.ones([X.shape[0],X.shape[1]+1])
# print(temp_X.shape)               ## (47, 2)
# print(temp_X[:5])                 
'''
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
'''
# temp_X = np.ones((X1.shape[0], 3))
# temp_X[:, 1:2] = X1
# temp_X[:, 2:3] = X2
# print(temp_X[:5])

temp_X[:,1:]=X
# print(temp_X[:5])
# print(temp_X.shape)               ## (47, 3)

theta= np.zeros((X.shape[1]+1,1))
# print(theta.shape)               ## (3, 1)

s_theta=np.zeros((X.shape[1]+1,1))   ##(3, 1)
# print(np.matmul(temp_X,theta).shape)       ## (47, 1)   ## matmul ==> give multiplication of matrix and mul

mb_theta=np.zeros((X.shape[1]+1,1))  

## HYPOTHESIS
def h(theta,X):
    tempX = np.ones((X.shape[0],X.shape[1]+1))
    tempX[:, 1:] = X
    return np.matmul(tempX, theta)

def gradient(theta,X,y):
    tempX = np.ones((X.shape[0],X.shape[1]+1))
    tempX[:, 1:] = X
    d_theta = np.average((h(theta,X)-y)*tempX, axis=0)
    # print(d_theta)
    # print(d_theta.shape) #(3,)
    d_theta = np.reshape(d_theta, (d_theta.shape[0],1))
    # print(d_theta.shape) #(3,1)
    return d_theta
    
def loss(theta,X,y):
    y_hat = h(theta,X)
    return np.average(np.square(y - y_hat))

## BATCH GRADIENT DESCENT
print("\n BATCH GRADIENT DESCENT")
def gradient_descent(theta,X,y,Learning_rate,max_itr,gap):
    cost = np.zeros(max_itr)

    for i in range(max_itr):
        d_theta = gradient(theta,X,y)
        theta = theta - Learning_rate * d_theta
        # print(theta.shape)
        # print(theta)
        cost[i] = loss(theta,X,y)
        if i % gap == 0:
            print("Iteration :",i, "\n Loss = ", loss(theta,X,y))
    return theta, cost

theta, cost = gradient_descent(theta,X,y,learning_rate,max_itr,100)
print("Final theta [Batch Gradient] :", theta)
# print("Final loss:", cost)

######################## Stochastic GD ########################
print("\n Stochistic Gradient Decent ")
def stochistic_gradient_decent(theta,X,y,learning_rate,max_itr,gap):
    cost = np.zeros(max_itr)

    for i in range(max_itr):
        for j in range(X.shape[0]):
            d_theta = gradient(theta,X[j,:].reshape(1,X.shape[1]),y[j,:].reshape(1,y.shape[1]))
            # print(X[j,:].reshape(1,X.shape[1]).shape)  # (1, 2)
            theta = theta - learning_rate*d_theta     
            # print(y[j,:].reshape(1,1))  # (1, 1)
        cost[i] = loss(theta,X,y)
        if i%gap==0:
            print("Iteration:",i,"| loss:",loss(theta,X,y))
    return theta, cost
        

s_theta,s_cost = stochistic_gradient_decent(s_theta,X,y,s_learning_rate,s_max_itr,100)
print("Final Theta [stochistic_gradient_decent]:",s_theta)
print("Final Loss:",loss(s_theta,X,y))
print("Cost :",s_cost)


######################### MINIBATCH GRADIENT DESCENT #########################
print("\n MINIBATCH GRADIENT DESCENT")
def mini_batch_gradient_decent(theta,X,y,learning_rate,max_itr,gap):
    cost=np.zeros(max_itr)
    for i in range(max_itr):
        for j in range(0,X.shape[0],batch_size):
            d_theta=gradient(theta,X[j:j+batch_size,:],y[j:j+batch_size,:])
            theta=theta-learning_rate*d_theta

        cost[i] = loss(theta,X,y)
        if i%gap==0:
            print("Iteration:",i,"| loss:",loss(theta,X,y))
    return  theta, cost

mb_theta,mb_cost=mini_batch_gradient_decent(mb_theta,X,y,mb_learning_rate,mb_max_itr,100)
print("Final theta [Batch Gradient] :", mb_theta)
print("Final loss:", mb_cost)

fig,ax=plt.subplots()
ax.plot(np.arange(max_itr),cost,'r')
ax.plot(np.arange(s_max_itr),s_cost,'b')
ax.plot(np.arange(mb_max_itr),mb_cost,'g')
ax.legend(loc='upper right',labels=['Batch gradient descent','Stochistic Gradient Decent','MINIBATCH GRADIENT DESCENT'])
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()



































