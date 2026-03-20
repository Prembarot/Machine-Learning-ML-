import numpy as np

list=np.array([10,20,30,40,50,60])
# print(list.shape)
# print(list)

##  ROW VECTOR 
x=list[np.newaxis,:]      #(ROWS,COLUMN)
print(x.shape)
print(x)

## COLUMN VECTOR 
y=list[:,np.newaxis]
print(y.shape)
print(y)