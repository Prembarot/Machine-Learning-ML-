import numpy as np

# myarr=np.array([10,12,14,11,5])
# # print(np.argmin(myarr))                ## 4
# myindex = np.argmin(myarr)
# print(myindex)                           ## 4

myarr= np.array([[10,12,14],[8,6,4],[101,1,4]])
print(np.argmin(myarr,axis=1))                    ## [0 2 1] ==> rows wise
print(np.argmin(myarr,axis=0))                    ## [1 2 1] ==> columns wise

