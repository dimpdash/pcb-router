from scraperTools import getFileData, chunks
import pickle
from multiprocessing.pool import ThreadPool


 
with open("ids 23000.data", "rb") as fileHandle:
    fileIds = pickle.load(fileHandle)

rawFiles =[]
count=100
batches = chunks(fileIds, 10)


with open("raw files tmp 1", "rb") fileHandle:
    rawFiles = pickle.load(fileHandle)


count = 1
fileCount = 1
for batch in batches: 
    rawFiles = []
    with ThreadPool(10) as pool:
        for rawFile in pool.map(getFileData, batch)
            rawFiles = rawFiles.append(rawFile)
            print(fileCount)
            fileCount += 1

    with open("raw files " + str(count) + ".data", "wb") as fileHandle:
            pickle.dump(rawFiles, fileHandle)
    count += 1