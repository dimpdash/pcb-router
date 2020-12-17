from scraperTools import seqSaveDataToFile, getDataFromFile, deleteOldFile
from createTrainingAndTestData import extractPads, extractRouteElements
import random

trainInputPath = "./model data/testInput.data"
trainOutputPath = "./model data/testOutput.data"
testInputPath = "./model data/trainInput.data"
testOutputPath = "./model data/trainOutput.data"

# Delete previous files test and train input output files
deleteOldFile(trainInputPath) #TODO
deleteOldFile(trainOutputPath)
deleteOldFile(testInputPath)
deleteOldFile(testOutputPath)

for i in range(0, 10 + 1):
    print(i)
    pcbs = getDataFromFile("./pcb files json/pcb files json " + str(i) + ".data")
   
    for pcb in pcbs:
        if random.randint(1,10) == 1: # testData
            testInput = extractPads(pcb)
            testOutput = extractRouteElements(pcb)
            seqSaveDataToFile(testInputPath, testInput)
            seqSaveDataToFile(testOutputPath, testOutput)
        else: #trainData
            trainInput = extractPads(pcb)
            trainOutput = extractRouteElements(pcb)
            seqSaveDataToFile(trainInputPath, trainInput)
            seqSaveDataToFile(trainOutputPath, trainOutput) 

