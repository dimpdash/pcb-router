import pickle
from scraperTools import getPCB_data

for i in range(1, 10 + 1):
    print(i)
    pcbDataFiles = []

    with open("./raw data files/raw files " + str(i) + ".data", "rb") as fileHandle:
        rawDataFiles = pickle.load(fileHandle)

    for rawData in rawDataFiles:
        try:
            pcbDataFiles.append(getPCB_data(rawData))
        except:
            pass
    
    with open("./pcb files/pcb files " + str(i) + ".data", "wb") as fileHandle:
        pickle.dump(pcbDataFiles, fileHandle)