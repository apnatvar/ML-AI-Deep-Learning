import copy

distanceMatrix = [[ 0, 1, 5, 15, 0],
                 [ 1, 0, 0, 0, 10],
                 [ 5, 0, 0, 0, 5],
                 [15, 0, 0, 0, 5],
                 [0, 10, 5, 5, 0]]
dict = { 0: "S", 1: "A", 2: "B", 3: "C", 4: "G" }

def costs(previousCost, sample):
    i = len(sample)-1
    newCost = previousCost + distanceMatrix[sample[i-1]][sample[i]]
    return newCost
            
start = (0,[ 0 ])
pathQueue = [ start ]
completed = 0
while completed < len(pathQueue):
    pathTuple = pathQueue[completed]
    path = pathTuple[1]
    last = path[len(path)-1]
    if last == 4:
        completed = completed + 1
        continue
    tempQueue = [ ]
    for i in range(5):
        if i not in path and distanceMatrix[last][i] != 0:
            tempPath = copy.deepcopy(path)
            tempPath.append(i)
            tempQueue.append((costs(pathTuple[0],tempPath),tempPath))
    tempQueue.sort()
    pathQueue.pop(completed)
    for i in tempQueue:
        pathQueue.append(i)
pathQueue.sort()
print("-------------------------------------------")
print("Paths and their costs in ascending order")
for i in pathQueue:
    for j in i[1]:
        print(dict.get(j))
    print("- " + str(i[0]))
print("-------------------------------------------")

'''import copy
import numpy as np
distanceMatrix = [[ 0, 1, 5, 15, 0],
                 [ 1, 0, 0, 0, 10],
                 [ 5, 0, 0, 0, 5],
                 [15, 0, 0, 0, 5],
                 [0, 10, 5, 5, 0]]
dict = { 0: "S", 1: "A", 2: "B", 3: "C", 4: "G" }

def costs(previousCost, sample):
    i = len(sample)-1
    newCost = previousCost + distanceMatrix[sample[i-1]][sample[i]]
    return newCost
            
start = (0,[ 0 ])
pathQueue = [ start ]
completed = 0
while completed < len(pathQueue):
    i = completed
    pathTuple = pathQueue[i]
    path = pathTuple[1]
    last = path[len(path)-1]
    if last == 4:
        completed = completed + 1
        #print(completed
        continue
    tempQueue = [ ]
    for index in range(5):
        if index not in path and distanceMatrix[last][index] != 0:
            tempPath = copy.deepcopy(path)
            tempPath.append(index)
            tempQueue.append((costs(pathTuple[0],tempPath),tempPath))
    tempQueue.sort(reverse=True)
    pathQueue.pop(i)
    for t in tempQueue:
        pathQueue.insert(i,t)
pathQueue.sort()
print("-------------------------------------------"
print("Paths and their costs in ascending order"
for i in pathQueue:
    for j in i[1]:
        print(dict.get(j),
    print("- " + str(i[0])
print("-------------------------------------------"*/'''


    
    
    


