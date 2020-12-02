import pickle
from scraperTools import getPCB_data, getDataFromFile, saveDataToFile
from PCB_parser import convertPCBcompressedToJSON

for i in range(0, 10 + 1):
    print(i)
    pcbs = getDataFromFile("./pcb files/pcb files " + str(i) + ".data")
    pcbsJSON = []
    for pcb in pcbs:
        pcbsJSON.append(convertPCBcompressedToJSON(pcb))
    saveDataToFile("./pcb files json/ pcb files json " + str(i) + ".data", pcbsJSON)
