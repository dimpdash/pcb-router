
from scraperTools import getProjectLinksOffpage
# load additional module
import pickle

pages = 953


projectLinks = []

with open("projectLinks200 - Copy.data", "rb") as filehandle:
    projectLinks = pickle.load(filehandle)

for page in range(201,pages+1):
    currentPagesLinks = getProjectLinksOffpage(page)
    projectLinks = projectLinks + currentPagesLinks
    print("Completed Page ", page, "List length", len(currentPagesLinks))

    if page % 200 == 0:
        with open('projectLinks' + str(page)+ '.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(projectLinks, filehandle)

with open('projectLinks' + str(pages+1)+ '.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(projectLinks, filehandle)