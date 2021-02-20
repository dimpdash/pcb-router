import numpy as np
from scraperTools import getDataFromFile
from tensorflow.keras.utils import to_categorical

ids = getDataFromFile("./model data/ids.data")

netNames = set()

for id in ids:
    pcb = np.load("./model data/"+id+".data")
    nets = [element['net'] for element in pcb]
    netNames = netNames.union(nets)

print(netNames)
print(len(netNames))

np.save('./model data/netNames.npy')