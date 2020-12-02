import requests
from bs4 import BeautifulSoup
import re
import json
import pickle
import sys

def getProjectLinksOffpage(page):

    URL = 'https://oshwlab.com/explore?projectSort=updatedTime&page=' + str(page)
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, 'html.parser')
    project_containers = soup.find_all("a", class_="project-link")

    projectLinks=[]
    for project in project_containers:
        projectLinks.append(project.get('href'))
    return projectLinks
   


def getFileIds(projectLink):
    projectPage = requests.get("https://oshwlab.com" + projectLink)
    projectPageSoup = BeautifulSoup(projectPage.content, 'html.parser')

    
    button = projectPageSoup.find("a", text="Open all in editor")
    if button is not None:
        fileLink = button.get('href')
        print(fileLink)
        ids = re.split(r'\||=',fileLink)
        ids = ids[1:]
        if '' in ids:
            ids.remove('')
    else:
        ids = []

    return ids

def getFileData(id):
    projectSourceAPI_URL = "https://easyeda.com/api/documents/" + id + "?version=6.4.7&uuid=" + id
    print(projectSourceAPI_URL)
    projectSource = requests.get(projectSourceAPI_URL)
    return projectSource.json()

def getPCB_data(text):
    if '"docType":3' not in text:
        return None
    
    start = text.index('shape":[') + len('shape":[')
    openCount = 0
    end = None
    for i in range(start, len(text)):
        if text[i] =="]":
            openCount -= 1
            if openCount == -1:
                end = i
                break
        elif text[i] == "[":
            openCount += 1
    if end == None:
        return None #file has mismatching [ ] pairs

    shapeData = text[start:end]

    jsonData = json.loads(text[:start-1] +'"tmp"'+ text[end+1:])
    if "result" in jsonData:
        if "dataStr" in jsonData["result"]:
            jsonData["result"]["dataStr"] = shapeData
            return jsonData
    return None

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def getDataFromFile(path):
    with open(path, 'rb') as fileHandle:
        return pickle.load(fileHandle)

def saveDataToFile(path, data):
    with open(path, "wb") as fileHandle:
        pickle.dump(data, fileHandle)