import pickle
from scraperTools import getPCB_data

def main():
    for i in range(0, 10 + 1):
        print(i)
        pcbDataFiles = []

        with open("./raw data files/raw files " + str(i) + ".data", "rb") as fileHandle:
            rawDataFiles = pickle.load(fileHandle)

        count = 0
        failCount = 0
        for rawData in rawDataFiles:
            print(count, i)
            if rawData["success"] == True:
                if rawData["result"]["docType"] == 3:
                    pcbDataFiles.append(rawData)
                count += 1
            else:
                failCount += 1
        with open("./pcb files/pcb files " + str(i) + ".data", "wb") as fileHandle:
            pickle.dump(pcbDataFiles, fileHandle)

        print(failCount)

main()