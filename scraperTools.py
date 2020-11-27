import requests
from bs4 import BeautifulSoup
import re
import json

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

    projectSourceSoup = BeautifulSoup(projectSource.content, 'html.parser')
    fileData = projectSourceSoup.get_text()
    return fileData

def getPCB_data(text):
    return json.loads(text)["result"]["dataStr"]["canvas"]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]