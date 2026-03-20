import numpy as np

price=[20,10,50,15,40]
item=["vadapav","dabeli","dosa","panipuri","momos"]

# x=zip(price,item)
# # print(list(x))

# money, food= zip(*x)

# money,food = zip(*sorted(zip(price,item)))

money,food = (list(t) for t in  zip(*sorted(zip(price,item))))
print("Money : ",money)
print("Food : ",food)