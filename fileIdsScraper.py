from scraperTools import getFileIds
# load additional module
import pickle



with open("projectLinks954.data", "rb") as fileHandle:
    projectLinks = pickle.load(fileHandle)


projectIds = []
with open("ids 23000.data", "rb") as fileHandle:
    projectIds = pickle.load(fileHandle)

count = 23000
for projectLink in projectLinks[count:]:
    print("Project number ", count)
    projectIds = projectIds + getFileIds(projectLink)

    # if count % 100 == 0:
    #     with open("ids tmp " + str((count // 100) % 10) + ".data", "wb") as fileHandle:
    #         pickle.dump(projectIds, fileHandle)
    # if count % 1000 == 0:
    #     with open("ids " + str(count) + ".data", "wb") as fileHandle:
    #         pickle.dump(projectIds, fileHandle)
    count += 1

with open("ids " + str(count) + ".data", "wb") as fileHandle:
            pickle.dump(projectIds, fileHandle)    