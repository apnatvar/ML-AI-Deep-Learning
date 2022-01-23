#submitted by
#Apnatva Singh Rawat
#101903655
#2COE25
def outMessage(vol):
    print("Volumes at this stage are")
    print(vol)
    print("-------------------------------")
    
def fill4(vol):
    newVol = [vol[0],4]
    print("-------------------------------")
    print("Filling 4")
    outMessage(newVol)
    return newVol
    
def fill3(vol):
    newVol = [3,vol[1]]
    print("-------------------------------")
    print("Filling 3")
    outMessage(newVol)
    return newVol
    
def empty4(vol):
    newVol = [vol[0],0]
    print("-------------------------------")
    print("Emptying 4")
    outMessage(newVol)
    return newVol
    
def empty3(vol):
    newVol = [0,vol[1]]
    print("-------------------------------")
    print("Emptying 3")
    outMessage(newVol)
    return newVol
    
def transfer4To3(vol):
    if vol[0] + vol[1] > 3:
        volTransferred = 3 - vol[0]
        newVol = [3, vol[1] - volTransferred]
    else:
        newVol = [vol[0] + vol[1], 0]
    print("-------------------------------")
    print("Transfering contents of 4 to 3")
    outMessage(newVol)
    return newVol
    
def transfer3To4(vol):
    if vol[0] + vol[1] > 4:
        volTransferred = 4 - vol[1]
        newVol = [vol[0] - volTransferred, 4]
    else:
        newVol = [0, vol[0] + vol[1]]
    print("-------------------------------")
    print("Transfering contents of 3 to 4")
    outMessage(newVol)
    return newVol
    
vol = [0,0]
print("-------------------------------")
print("Volumes initially are")
print(vol)
print("-------------------------------")
while 1:
    if vol[0] == 0:
        vol = fill3(vol)
    elif vol[1] == 0:
        vol = transfer3To4(vol)
    elif vol[0] == 3 and vol[1] == 4:
        vol = empty4(vol)
        vol = transfer3To4(vol)
    elif vol[0] == 3 and vol[1] != 4:
        vol = transfer3To4(vol)
        vol = empty4(vol)
    elif vol[1] == 4:
        vol = empty4(vol)
    if vol[0] == 2:
        vol = empty4(vol)
        vol = transfer3To4(vol)
    if vol[1] == 2:
        print("We have obtained 2 litres in 4 litre jug")
        print("-------------------------------")
        break
#submitted by
#Apnatva Singh Rawat
#101903655
#2COE25
