import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
myData = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])
myColNames = ['Temperature', 'Activity']
myDataFrame = pd.DataFrame(data=myData, columns=myColNames)
print(myDataFrame)
myDataFrame["Adjusted"] = myDataFrame["Activity"] + 2
print(myDataFrame)
print("Rows #0, #1, and #2:")
print(myDataFrame.head(3), '\n')
print("Row #2:")
print(myDataFrame.iloc[[2]], '\n')
print("Rows #1, #2, and #3:")
print(myDataFrame[1:4], '\n')
print("Column 'Temperature':")
print(myDataFrame['Temperature'])
print(myDataFrame.corr())
print(myDataFrame.describe())
print(myDataFrame.head(n=1000))
print(myDataFrame.shape)
print(myDataFrame.dtypes)
myDataFrame.hist(bins = 10)
plt.show()
#histogram = myDataFrame.hist(column="Temperature")
