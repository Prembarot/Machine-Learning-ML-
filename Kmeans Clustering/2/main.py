import numpy as np
from  dataset.iris_dataset import IrisDataset
from Models.Clustering.kmeans_clustering import KmeansClustering
import matplotlib.pyplot as plt

IRIS_DIR_PATH = "Data"
queries = ["Species == 'Iris-setosa' | Species == 'Iris-versicolor' | Species == 'Iris-virginica'"]
features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
labels=["Species"]

def plot_clusters_centroids(X_train,clusters,centroids,x=0,y=0):
    colors=["r","g","b","y"]
    for i in range(centroids.shape[0]):     ## centroids.shape ==> k x 4
        data=X_train[clusters == i]
        plt.scatter(data[:,x],data[:,y],c=colors[i],s=20)
    plt.scatter(centroids[:,x],centroids[:,y],marker="*",s=100,c="#012257")
    plt.show()
        
def main():
    (X_train,y_train),(X_test,y_test)=IrisDataset.load_data(IRIS_DIR_PATH,queries,features,labels,scale=True)
    
    k=3
    model=KmeansClustering(k=3)
    model.fit(X_train)
    plot_clusters_centroids(X_train,model.clusters,model.centroids,x=0,y=1)
    plot_clusters_centroids(X_train,model.clusters,model.centroids,x=2,y=3)
    plot_clusters_centroids(X_train,model.clusters,model.centroids,x=2,y=1)
 
if __name__ =='__main__':
    main()
    
