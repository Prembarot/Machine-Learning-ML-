import numpy as np

home=[10,20,30,40]
jugnu=[1,2,3,4,5,6,7,8,9,10]

dist_array=[]

for h in home:           ## 4 time loop work
    for j in jugnu:      ## 10 time loop work
        dist=np.square(h-j)
        dist_array.append(dist)
        
print(sorted(dist_array))
    
