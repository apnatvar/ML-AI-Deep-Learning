import copy
import numpy as np

initial = [[2,8,3],[1,5,4],[7,6,0]]
goal = [[1,2,3],[8,0,4],[7,6,5]]

#initial = [[2,0,3],[1,8,4],[7,6,5]]
#goal = [[1,2,3],[8,0,4],[7,6,5]]

def heuristic(currentState):
    moves = 0
    for i in range(2):
        for j in range(2):
            if (currentState[i][j] != goal[i][j]):
                moves = moves + 1
    return moves
    
def compareHeuristic(parent,child):
    if child < parent:
        return True
    else:
        return False
        
def moveUp( puzzle, indexR, indexC):
    newIndexR = indexR - 1
    modifiedPuzzle = copy.deepcopy(puzzle)
    modifiedPuzzle[indexR][indexC],modifiedPuzzle[newIndexR][indexC] = modifiedPuzzle[newIndexR][indexC], modifiedPuzzle[indexR][indexC]
    return modifiedPuzzle

def moveDown( puzzle, indexR, indexC):
    newIndexR = indexR + 1
    modifiedPuzzle = copy.deepcopy(puzzle)
    modifiedPuzzle[indexR][indexC],modifiedPuzzle[newIndexR][indexC] = modifiedPuzzle[newIndexR][indexC], modifiedPuzzle[indexR][indexC]
    return modifiedPuzzle
    
def moveLeft( puzzle, indexR, indexC):
    newIndexC = indexC - 1
    modifiedPuzzle = copy.deepcopy(puzzle)
    modifiedPuzzle[indexR][indexC],modifiedPuzzle[indexR][newIndexC] = modifiedPuzzle[indexR][newIndexC], modifiedPuzzle[indexR][indexC]
    return modifiedPuzzle
    
def moveRight( puzzle, indexR, indexC):
    newIndexC = indexC + 1
    modifiedPuzzle = copy.deepcopy(puzzle)
    modifiedPuzzle[indexR][indexC],modifiedPuzzle[indexR][newIndexC] = modifiedPuzzle[indexR][newIndexC], modifiedPuzzle[indexR][indexC]
    return modifiedPuzzle

parentQueue = [ initial ]
childQueue = [ initial ]
parentCopy = initial
for parent in childQueue:
    a = np.array(parent)
    R, C = np.where(a == 0)
    indexR = R[0]
    indexC = C[0]
    if indexR != 0:
        up = moveUp(parent, indexR, indexC)
        if up == goal:
            parentCopy = up
            parentQueue.append(parent)
            childQueue.append(up)
            break
        else:
            if compareHeuristic(heuristic(parent),heuristic(up)):
                parentQueue.append(parent)
                childQueue.append(up)
                
    if indexR != 2:
        down = moveDown(parent, indexR, indexC)
        if down == goal:
            parentCopy = down
            parentQueue.append(parent)
            childQueue.append(down)
            break
        else:
            if compareHeuristic(heuristic(parent),heuristic(down)):
                parentQueue.append(parent)
                childQueue.append(down)
                
    if indexC != 0:
        left = moveLeft(parent, indexR, indexC)
        if left == goal:
            parentCopy = left
            parentQueue.append(parent)
            childQueue.append(left)
            break
        else:
            if compareHeuristic(heuristic(parent),heuristic(left)):
                parentQueue.append(parent)
                childQueue.append(left)
                
    if indexC != 2:
        right = moveRight(parent, indexR, indexC)
        if right == goal:
            parentCopy = right
            parentQueue.append(parent)
            childQueue.append(right)
            break
        else:
            if compareHeuristic(heuristic(parent),heuristic(right)):
                parentQueue.append(parent)
                childQueue.append(right)
solutionAchieved = copy.deepcopy(parentCopy)
solutionPath = [ parentCopy ]
while parentCopy != initial:
    index = 0
    for i in childQueue:
        if i == parentCopy:
            break
        else:
            index = index + 1
    parentCopy = parentQueue[index]
    solutionPath.append(parentCopy)
for i in range(len(solutionPath)):
    solutionPath.append(solutionPath.pop(len(solutionPath)-i-1))
print("--------------------------------------------------------")
print("Solution achieved is: " + str(solutionAchieved))
print("Solution was achieved in " + str(len(solutionPath)-1) + " moves")
print("Path followed for the solution is:")
for i in solutionPath:
    print(i)
print("--------------------------------------------------------")
#submitted by
#Apnatva Singh Rawt
#101903655
#2COE25

