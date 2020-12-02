import pickle
from scraperTools import getPCB_data

for i in range(0, 10 + 1):
    print(i)
    pcbDataFiles = []

    with open("./raw data files/raw files " + str(i) + ".data", "rb") as fileHandle:
        rawDataFiles = pickle.load(fileHandle)

    count = 0
    for rawData in rawDataFiles:
        print(count, i)
        pcbData = getPCB_data(rawData)
        if pcbData != None:
            pcbDataFiles.append(pcbData)
        count += 1
    with open("./pcb files/pcb files " + str(i) + ".data", "wb") as fileHandle:
        pickle.dump(pcbDataFiles, fileHandle)