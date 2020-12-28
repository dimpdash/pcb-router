import random
from scraperTools import getDataFromFile, saveDataToFile
from tensorflow.keras.preprocessing.text import one_hot as one_hot_text
from tensorflow import one_hot
import numpy as np

random.seed("a")

net_vocab_size = 250
max_net_char_len = 8
maxLayers = 32 + 2

def extractPads(shapes):
    padsList = []
    for shape in shapes:
        if "PAD" in shape:
            padsList.append(compressPadData(shape["PAD"]))
        elif "LIB" in shape:
            padsList = padsList + extractPads(shape["LIB"])

    return padsList

def compressPadData(pad):
    
    # padTypes = {"RECT": [1, 0, 0],"OVAL":[0, 1, 0], "ELLIPSE":[0, 0, 1], "POLYGON": [0, 0, 0]}
    padCompressed = dict() 
    padCompressed["pos"] = [pad['x'],pad['y']]
    padCompressed["size"] = [pad['width'], pad['height']]
    padCompressed["layer"] = pad['layer']
    padCompressed["net"] = pad['net']
    # padCompressed[4] = padTypes[pad["type"]]
    # padCompressed[5] = pad['rot']

    return padCompressed

def extractRouteElements(shapes):
    routeList = []
    copperLayers = [1,2] + list(range(21,52 + 1))
    for shape in shapes:
        if "TRACK" in shape:
            if int(shape["TRACK"]["layer"]) in copperLayers:
                routeList = routeList + compressRouteData(shape["TRACK"])
        # elif "VIA" in shape: # TODO Consider VIAs 
        #     routeList.append(compressRouteData(shape["VIA"]))
        elif "LIB" in shape:
            routeList = routeList + extractRouteElements(shape["LIB"])

    return routeList

def compressRouteData(route):
    #assumes route is track
    routeList = []
    lastPoint = route["points"][0]
    for point in route["points"][1:]: #takes each pair of points, so skips first
        routeCompressed = [None]*7
        routeCompressed[0] = route["width"]
        routeCompressed[1] = route["layer"]
        routeCompressed[2] = route["net"]
        routeCompressed[3] = point[0]
        routeCompressed[4] = point[1]
        routeCompressed[5] = lastPoint[0]
        routeCompressed[6] = lastPoint[1]        

        routeList.append(routeCompressed)
        lastPoint = point

    return routeList

def getId(pcb):
    
