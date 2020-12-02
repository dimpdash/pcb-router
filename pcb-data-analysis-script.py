from scraperTools import getDataFromFile

pcbs = getDataFromFile("./pcb files/pcb files 1.data")

pcb = pcbs[0]["shape"]
print(pcb)