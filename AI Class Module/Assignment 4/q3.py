import copy
startState = [ ['a'], ['b','c'], [ ] ]
goalState = ['a','b','c']

def move(toPop, toAdd, state):
    tempNewState = copy.deepcopy(state)
    tempNewState[toAdd].append( tempNewState[toPop].pop())
    return tempNewState

def checkDuplicate(currentState,allStates):
    for singleState in allStates:
            count = 0
            for i in range(3):
                for j in range(3):
                    if singleState[i] == currentState[j]:
                        count = count + 1
            if count == 3:
                return False
    return True
    
def checkGoal(state):
    for i in state:
        if i == goalState:
            print("------------------------------------")
            print(state)
            print("Reached at depth: " + str(currentDepth + 1))
            print("Exiting....")
            print("------------------------------------")
            quit()

indices = [ 0, 1, 2]
depthFinal = int(input("Enter the required depth to search: "))
states = [ ]
currentDepth = 0

for currentDepthFinal in range(0,depthFinal):
    currentDepth = 0
    currentStack = [ startState ]
    states = [ ]
    
    while currentDepth != currentDepthFinal and currentStack != []:
        currentDepth = currentDepth + 1
        state = currentStack.pop()
        for i in range(3):
            indices.pop(indices.index(i))
            if state[i] != []:
                for j in indices:
                    newState = move(i,j,state)
                    if checkDuplicate(newState,states):
                        #print(newState
                        currentStack.append(newState)
            indices.append(i)
        states.append(state)
        
    for state in states:
        checkGoal(state)
            
print("-------------------------")
print("Not Found")
print("-------------------------")
