import numpy as np

a = ["cheese", "paneer", "Dal"]
b = ["Burger", "Roll", "Rice"]

x = list(zip(a, b))
# print(x)

topping, base = zip(*x)
print("Toppings are :", topping)
print("Base are :", base)
