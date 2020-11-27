import pickle
rawFiles = []
for i in range(1,704):
    print(i)
    with open("raw files " + str(i) + ".data", "rb") as fileHandle:
        rawFiles = rawFiles + pickle.load(fileHandle)

with open("raw files 0.data", "wb") as fileHandle:
    pickle.dump(rawFiles,fileHandle)



# with open("raw files 704.data", "rb") as filehandle:
#     a = pickle.load(filehandle)
#     print(a[1][0:100])


# with open("ids 23819.data", "rb") as filehandle:
#     a = pickle.load(filehandle)
#     print(len(a))

