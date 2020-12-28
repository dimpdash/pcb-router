import pickle
from scraperTools import getPCB_data, getDataFromFile, saveDataToFile
from PCB_parser import convertPCBcompressedToJSON

for i in range(0, 10 + 1):
    print(i)
    pcbs = getDataFromFile("./pcb files/pcb files " + str(i) + ".data")
    pcbsJSON = dict()
    ids = []
    for pcb in pcbs:
        pcbsJSON[pcb["result"]["uuid"]] = convertPCBcompressedToJSON(pcb)
        ids.append(pcb["result"]["uuid"])
    saveDataToFile("./pcb files json/pcb files json " + str(i) + ".data", pcbsJSON)
saveDataToFile("./pcb files json/ids.data")
