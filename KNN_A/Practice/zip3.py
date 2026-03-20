import numpy as np

names = ["Ravi", "Anita", "Karan", "Meena"]
marks = [78, 92, 85, 88]

sorted_name , sorted_marks = (list(t) for t in  zip(*sorted(zip(names,marks))))
print("sortedname : ",sorted_name)
print("sortedmarks : ",sorted_marks)