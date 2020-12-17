import random
from scraperTools import getDataFromFile, saveDataToFile
random.seed("a")


def extractPads(shapes):
    padsList = []
    for shape in shapes:
        if "PAD" in shape:
            padsList.append(compressPadData(shape["PAD"]))
        elif "LIB" in shape:
            padsList = padsList + extractPads(shape["LIB"])

    return padsList

def compressPadData(pad):
    
    padTypes = {"RECT": b'100',"OVAL":b'010', "ELLIPSE":b'001', "POLYGON": b'000'}
    padCompressed = [None]*8 
    
    padCompressed[0] = padTypes[pad["type"]]
    padCompressed[1] = pad['x']
    padCompressed[2] = pad['y']
    padCompressed[3] = pad['width']
    padCompressed[4] = pad['height']
    padCompressed[5] = pad['layer']
    padCompressed[6] = pad['net']
    padCompressed[7] = pad['rot']

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


