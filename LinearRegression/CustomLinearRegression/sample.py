import numpy as np

a=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
a=np.array(a)
# print(a.shape)   ## (15,)
a=a.reshape(a.shape[0],1)
# print(a.shape)   ## (15, 1)

for i in range(0,a.shape[0],3):             ## 0 3 6 9 12 
    # print(a[i],end="")                  ##[1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]
    print(i,end=" ")                       ## 01234567891011121314
print("\n")
