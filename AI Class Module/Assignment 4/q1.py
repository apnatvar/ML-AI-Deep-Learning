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
            print("-------------------------")
            print(state)
            print("Goal State Reached")
            print("Exiting....")
            print("-------------------------")
            #print(states)
            quit()

currentStack = [ startState ]
states = [ ]
indices = [ 0, 1, 2]
while currentStack != []:
    state = currentStack.pop()
    for i in range(3):
        indices.pop(indices.index(i))
        if state[i] != []:
            for j in indices:
                newState = move(i,j,state)
                if checkDuplicate(newState,states):
                    checkGoal(newState)
                    #print(newState
                    currentStack.append(newState)
        indices.append(i)
    states.append(state)


