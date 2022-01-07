import numpy as np
import random
oneDArray = np.array([1.2,2.4,3.5])
print(oneDArray)
twoDArray = np.array([[1,2],[3,4],[5,6]])
print(twoDArray)
randIntArray = np.random.randint(low=0, high=11, size=(6))
print(randIntArray)
randFloatArray = np.random.random([6])
print(randFloatArray)
randFloatArrayModded1 = randFloatArray + 2.0
print(randFloatArrayModded1)
randFloatArrayModded2 = randFloatArray * 3.0
print(randFloatArrayModded2)