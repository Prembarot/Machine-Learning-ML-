import numpy as np

products = ["Pen", "Book", "Bag", "Pencil"]
prices = [20, 120, 450, 10]

sorted_name, sorted_marks = ( list(t) for t in zip(*sorted(zip(products, prices), reverse=True)))

print("sortedname :", sorted_name)
print("sortedmarks :", sorted_marks)