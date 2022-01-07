import numpy as np
import random
feature = np.random.randint(low=6,high=21,size=(15))
print("feature: ",feature)
noise = np.random.random([15]) * 4 - 2
label = 3 * feature + 4
label = label + noise
print("noise: ",noise)
print("label: ",label)
