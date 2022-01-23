#submitted by
#Apnatva Singh Rawat
#101903655
#2COE25
import numpy as np
import random
distanceMatrix = [ [0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0] ]
startNode = int(input("Enter the starting node - " ))
startNode = startNode - 1

def travel(startNode):
    travelQueue = [ startNode ]
    travelled = 0
    while travelled != 3:
        possibleMove = random.randint(0,3)
        if possibleMove not in travelQueue:
            travelQueue.append(possibleMove)
            travelled = travelled + 1
    return travelQueue
    
def weights(startNode, samples, distanceMatrix):
    weightQueue = [ ]
    for sampleMove in samples:
        weight = 0
        for i in range(0,2):
            weight = weight + distanceMatrix[sampleMove[i]][sampleMove[i+1]]
        weightQueue.append(weight)
    return weightQueue
        
def possiblePaths (numberPossibleTravel, startNode):
    travelPaths = [ ]
    numberPossibleTravel = 6
    while numberPossibleTravel != 0 :
        possibleTravel = travel(startNode)
        if possibleTravel not in travelPaths:
            travelPaths.append(possibleTravel)
            numberPossibleTravel = numberPossibleTravel - 1
    return travelPaths

travelQueues = np.array(possiblePaths(6, startNode))
weightOfTravel = np.array(weights(startNode, travelQueues, distanceMatrix))
travelQueues = travelQueues + 1
print("-----------------------------")
print("Possible Paths are:")
print(travelQueues)
print("-----------------------------")
print("-----------------------------")
print("Shortest Path is:")
print(travelQueues[np.where( weightOfTravel == np.amin(weightOfTravel ))])
print("-----------------------------")
print("-----------------------------")
print("Length of Shortest Path is:")
print(np.amin(weightOfTravel))
print("-----------------------------")
#submitted by
#Apnatva Singh Rawat
#101903655
#2COE25
        
    
