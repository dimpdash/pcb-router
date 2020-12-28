from scraperTools import seqSaveDataToFile, getDataFromFile, deleteOldFile
from createTrainingAndTestData import extractPads, extractRouteElements, getId
import random

trainInputPath = "./model data/trainPads.data"
trainOutputPath = "./model data/trainRouting.data"

# Delete previous files test and train input output files
deleteOldFile(trainInputPath) #TODO
deleteOldFile(trainOutputPath)

for i in range(0, 10 + 1):
    print(i)
    pcbs = getDataFromFile("./pcb files json/pcb files json " + str(i) + ".data")
   
    for pcb in pcbs:
        trainInput = extractPads(pcb)
        trainOutput = extractRouteElements(pcb)
        id = getId(pcb)
        seqSaveDataToFile(trainInputPath, trainInput)
        seqSaveDataToFile(trainOutputPath, trainOutput) 

