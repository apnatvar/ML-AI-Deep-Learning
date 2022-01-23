import copy
import numpy as np

initial = [[2,0,3],[1,8,4],[7,6,5]]
goal = [[1,2,3],[8,0,4],[7,6,5]]

def heuristic(before, after, previousCost):
    move = 0
    for i in range(2):
        for j in range(2):
            if (before[i][j] != goal[i][j]):
                move = move + 1
    return (move + previousCost)
    
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
    
solutionMatrices = [ (heuristic(initial,goal, 0),[[2,0,3],[1,8,4],[7,6,5]]) ]
parentQueue = [ initial ]
finalAnswer =  [ ]
for parent in solutionMatrices:
    a = np.array(parent[1])
    R, C = np.where(a == 0)
    indexR = R[0]
    indexC = C[0]
    tempChildQueue = []
    if indexR != 0:
        up = moveUp( parent[1], indexR, indexC)
        temp = (heuristic(up, goal, parent[0]),up)
        if up == goal:
            finalAnswer = up
            solutionMatrices.append(temp)
            parentQueue.append(parent[1])
            break
        elif temp not in solutionMatrices:
            tempChildQueue.append(temp)
            
    if indexR != 2:
        down = moveDown( parent[1], indexR, indexC)
        temp = (heuristic(down, goal, parent[0]),down)
        if down == goal:
            finalAnswer = down
            solutionMatrices.append(temp)
            parentQueue.append(parent[1])
            break
        elif temp not in solutionMatrices:
            tempChildQueue.append(temp)
            
    if indexC != 0:
        left = moveLeft( parent[1], indexR, indexC)
        temp = (heuristic(left, goal, parent[0]),left)
        if left == goal:
            finalAnswer = left
            solutionMatrices.append(temp)
            parentQueue.append(parent[1])
            break
        elif temp not in solutionMatrices:
            tempChildQueue.append(temp)
            
    if indexC != 2:
        right = moveRight( parent[1], indexR, indexC)
        temp = (heuristic(right, goal, parent[0]),right)
        if right == goal:
            finalAnswer = right
            solutionMatrices.append(temp)
            parentQueue.append(parent[1])
            break
        elif temp not in solutionMatrices:
            tempChildQueue.append(temp)
            
    tempChildQueue.sort()
    for i in tempChildQueue:
        solutionMatrices.append(i)
        parentQueue.append(parent[1])

solutionPath = [ finalAnswer ]
solutionAchieved = copy.deepcopy(finalAnswer)

while finalAnswer != initial:
    index = 0
    for i in solutionMatrices:
        if i[1] == finalAnswer:
            break
        else:
            index = index + 1
    finalAnswer = parentQueue[index]
    solutionPath.append(finalAnswer)
    
for i in range(len(solutionPath)):
    solutionPath.append(solutionPath.pop(len(solutionPath)-i-1))
    
print("--------------------------------------------------------")
print("Solution achieved is: " + str(solutionAchieved))
print("Solution was achieved in " + str(len(solutionPath)-1) + " moves")
print("Path followed for the solution is:")
for i in solutionPath:
    print(i)
print("--------------------------------------------------------")
