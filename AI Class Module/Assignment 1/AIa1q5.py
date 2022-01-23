D = {1 :"One", 2: "Two", 3:"Three", 4: "Four", 5:"Five"}
D[6] = "Six"
del(D[2])
print D
if 6 in D:
    print ("exists")
else:
    print ("does not exist")
elements = 0
for i in D:
    elements = elements + 1
print ("Number of elements is " + str(elements))
