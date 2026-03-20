import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Data = []
for i in range(1,11):
    Data.append(i)
print("Data: ", Data)

train,test= train_test_split(Data,test_size=0.3)
print("Train: ", train)
print("Test: ", test)

# Split with random_state=30
train_10,test_10= train_test_split(Data,test_size=0.3,random_state=10)
print("No Random State ")
print("train_10",train_10)
print("test_10",test_10)

# Split with raddom_satet=42
train_42,test_42= train_test_split(Data,test_size=0.3,random_state=42)
print("Random State ")
print("train_42",train_42)
print("test_42",test_42)











    
