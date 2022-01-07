import pandas as pd
import numpy as np
import random

myColumns = ['Eleanor','Chidi', 'Tahani', 'Jason']
myData = np.random.randint(low=0, high=101, size=(4,4))
myDataFrame = pd.DataFrame(data=myData, columns=myColumns)
print(myDataFrame)
print(myDataFrame['Eleanor'][1])
myDataFrame['Janet'] = myDataFrame['Jason'] + myDataFrame['Tahani']
print(myDataFrame)
referenceOfDataFrame = myDataFrame
print(referenceOfDataFrame)
copyOfDataFrame = myDataFrame.copy() #true copy of the original dataframe
print(copyOfDataFrame)