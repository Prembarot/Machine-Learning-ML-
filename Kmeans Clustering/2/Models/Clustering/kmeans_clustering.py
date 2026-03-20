import numpy as np
from copy import deepcopy

class KmeansClustering:
    def __init__(self,k):
        self.k=k
        self.error=np.inf
        self.X=None
        self.centroids=None
        self.clusters=None
        
    def fit(self, X,max_itr=20,error_margin=0.001):
        # print("shape of X_train",X.shape)           ## (112, 4)
        self.X=X
        self.initilize_centroid()
        i = 1
        while i <= max_itr and self.error > error_margin:
            self.assign_clusters()
            self.realign_clentroids()
            i += 1
        
    def assign_clusters(self):
        distances=np.zeros(shape=(self.X.shape[0],self.k))
        for i in range(self.k):
            centroid = self.centroids[i,:]
            ## centroids.shape ==> 3x4
            ## centroid.shape ==> 1x4
            distances[:,i]=np.sum(np.square(self.X-centroid),axis=1)    ### axis=1 means row wise  and axis=0 means colum wise   
        self.clusters = np.argmin(distances,axis=1)
    
    def realign_clentroids(self):
        centroids_old=deepcopy(self.centroids)    ## N x 1
        for i in range (self.k):
            self.centroids[i,:] = np.mean(self.X[self.clusters == i],axis=0)
        self.error = np.average(np.square(self.centroids-centroids_old))
          
              
    def initilize_centroid(self):
        # print(self.X.shape)                   ## (112, 4)
        indices=np.random.randint(self.X.shape[0],size=self.k)
        # print(indices)      ## [95 87 80]
        self.centroids=self.X[indices,:]
        # print(self.centroids)             
        
        
        '''
            X.shape ==> N(rows) x F(features)
            centrioid.shape ==> K x F
            K is a number of centroids  

        '''
        