from collections import Counter

mylist = ['a','a','a','b','b','c']

# print(len(mylist))

# print(mylist.count('a'))
# print(mylist.count('b'))
# print(mylist.count('c'))

d = Counter(mylist)

# print(d)                       ## Counter({'a': 3, 'b': 2, 'c': 1})
# print(d.most_common())         ## [('a', 3), ('b', 2), ('c', 1)]
# print(d.most_common())
# print(d.most_common()[0])
# print(d.most_common()[1])
# print(d.most_common()[2])

# print(d.most_common()[0][0])    ## a
# print(d.most_common()[2][0])    ## C
# print(d.most_common()[1][1])    ## 2
# print(d.most_common()[2][1])    ## 1

# print(d.most_common(1))           ## [('a', 3)]
# print(d.most_common(2))           ## [('a', 3), ('b', 2)]
# print(d.most_common(3))           ## [('a', 3), ('b', 2), ('c', 1)]
# print(d.most_common(1)[0])        ## ('a', 3)
# print(d.most_common(1)[0][0])     ## a

print(d.most_common(1)[0][1])       ## 3




