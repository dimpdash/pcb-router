from scraperTools import getDataFromFile, saveDataToFileInd
from createTrainingAndTestData import extractPads, extractRouteElements
import numpy as np

trainInputPath = "./model data/pads/"
trainOutputPath = "./model data/routes/"
fileType = ".npy"

for i in range(0, 10 + 1):
    print(i)
    pcbs = getDataFromFile("./pcb files json/pcb files json " + str(i) + ".data")
   
    for id in pcbs:
        pcb = pcbs[id]
        np.save(trainInputPath + str(id), extractPads(pcb))
        np.save(trainOutputPath + str(id), extractRouteElements(pcb))