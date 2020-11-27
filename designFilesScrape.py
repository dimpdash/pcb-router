from scraperTools import getFileData, chunks
import pickle
from multiprocessing.pool import ThreadPool

with open("ids 23819.data", "rb") as fileHandle:
    fileIds = pickle.load(fileHandle)

batches = chunks(fileIds[7040:], 5000)
count = 1
fileCount = 1
for batch in batches: 
    rawFiles = []
    with ThreadPool(10) as pool:
        for rawFile in pool.map(getFileData, batch):
            rawFiles.append(rawFile)
            print(fileCount)
            fileCount += 1

    with open("raw files " + str(count) + ".data", "wb") as fileHandle:
            pickle.dump(rawFiles, fileHandle)
    count += 1