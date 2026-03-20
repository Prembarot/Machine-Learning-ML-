import numpy as np

print(np.random.randint(9))
print(np.random.randint(9))
print(np.random.randint(9))
print(np.random.randint(9))
print(np.random.randint(9))
print(np.random.randint(9))
print(np.random.randint(9))
print(np.random.randint(9))
print(np.random.randint(9))
print(np.random.randint(9))

print(np.random.randint(2,size=10))          ## [1 0 0 1 1 1 0 0 0 0]

print(np.random.randint(15,size=10))         ## [ 3 14 10  1  4  9  3 11  4  1]

print(np.random.randint(15,size=(3,5)))
'''
    [[ 9  4  8  6  3]
     [11 10 12  4 11]
    [ 9 10  7  3  4]]

'''

print(np.random.randint(10,size=(5,2)))
'''
        [[1 5]
        [8 6]
        [7 5]
        [5 3]
        [7 6]]

'''
