#submitted by
#Apnatva Singh Rawat
#101903655
#2COE25
import numpy as np
import copy

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

x = [ [ [1,2,3], [8,0,4], [7,6,5] ] ]
y = [ [2,8,1], [0,4,3], [7,6,5] ]
solutionMatrices = x
finalSolution = y
for solution in solutionMatrices:
    a = np.array(solution)
    indexR,indexC = np.where(a == 0)
    #print(indexR,"",indexC)
    #print( "------------------------------------" )
    #print( solution
    #print( "------------------------------------" )
    if indexR != 0:
        up = moveUp( solution, indexR[0], indexC[0])
        if up == finalSolution:
            print("-----------------------------------" )
            print(up)
            print("Solved")
            print("-----------------------------------")
            break
        elif up not in solutionMatrices:
            solutionMatrices.append(up)
            
    if indexR != 2:
        down = moveDown( solution, indexR[0], indexC[0])
        if down == finalSolution:
            print("-----------------------------------")
            print(down)
            print("Solved")
            print("-----------------------------------")
            break
        elif down not in solutionMatrices:
            solutionMatrices.append(down)
            
    if indexC != 0:
        left = moveLeft( solution, indexR[0], indexC[0])
        if left == finalSolution:
            print("-----------------------------------")
            print(left)
            print("Solved")
            print("-----------------------------------")
            break
        elif left not in solutionMatrices:
            solutionMatrices.append(left)
            
    if indexC != 2:
        right = moveRight( solution, indexR[0], indexC[0])
        if right == finalSolution:
            print("-----------------------------------")
            print(right)
            print("Solved")
            print("-----------------------------------")
            break
        elif right not in solutionMatrices:
            solutionMatrices.append(right)
#print( "------------------------------------"
#print( solutionMatrices
#print( "------------------------------------"
#submitted by
#Apnatva Singh Rawat
#101903655
#2COE25
