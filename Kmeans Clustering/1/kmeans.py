import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('Data//income.csv')
# print(df[:5])
# print(df.shape)

# X=df[['Name','Age']]            
# y=df['Income($)']                  
# print(X.shape)                  ## (22, 2)
# print(y.shape)                  ## (22,)

scaler=MinMaxScaler()
df[['Age','Income($)']] = scaler.fit_transform(df[['Age','Income($)']])
# print(df[['Age','Income($)']][:5])

kmeans= KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(df[['Age','Income($)']])
# print(y_pred)

df['Cluster']=y_pred
# print(df)

# print(kmeans.cluster_centers_)

# plt.scatter(df['Age'],df['Income($)'])
# plt.xlabel("AGE")
# plt.ylabel("INCOME ($)")
# plt.show()

df1 = df[df.Cluster == 0]
# print(df1)
df2 = df[df.Cluster == 1]
# print(df2)
df3 = df[df.Cluster == 2]
# print(df3)

# plt.scatter(df1.Age, df1['Income($)'], color='green',label="Cluster-1")
# plt.scatter(df2.Age, df2['Income($)'], color='red',label="Cluster-2")
# plt.scatter(df3.Age, df3['Income($)'], color='blue',label="Cluster-3")
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='yellow',marker='*',label="Centroid")
# plt.xlabel("Age")
# plt.ylabel("Income ($)")
# plt.legend()
# plt.show()


### sse ==> sum of square error
# print(kmeans.inertia_)                ## 0.4750783498553096

sse=[]
for k in range(1,10):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(df[['Age','Income($)']])
    sse.append(kmeans.inertia_)
# print(sse)

k_range= range(1,10)
plt.xlabel("K")
plt.ylabel("sum of square error")
plt.plot(k_range,sse)
plt.show()