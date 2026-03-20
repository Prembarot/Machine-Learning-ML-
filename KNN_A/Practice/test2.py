import numpy as np
 
home= np.array([[10,20,30,40]])          ## (1, 4)
jugnu=np.array([[1,2,3,4]])              ## (1, 4)
# print(home.shape)
# print(jugnu.shape)

print(home-jugnu)                        ## [[ 9 18 27 36]]
print(np.square(home-jugnu))             ## [[  81  324  729 1296]]
print(np.sum(np.square(home-jugnu)))     ## 2430

