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

    for line in route["points"]:
        routeCompressed = [None]*5
        routeCompressed[0] = route["width"]
        routeCompressed[1] = route["layer"]
        routeCompressed[2] = route["net"]
        routeCompressed[3] = line[0]
        routeCompressed[4] = line[0]

        routeList.append(routeCompressed)

    return routeList

trainInputs = []
testInputs = []
trainOutputs = []
testOutputs = []

for i in range(0, 10 + 1):
    print(i)
    pcbs = getDataFromFile("./pcb files json/pcb files json " + str(i) + ".data")
   
    for pcb in pcbs:
        if random.randint(1,10) == 1: # testData
            testInputs.append(extractPads(pcb))
            testOutputs.append(extractRouteElements(pcb))
        else: #trainData
            trainInputs.append(extractPads(pcb))
            trainOutputs.append(extractRouteElements(pcb))


trainData = [trainInputs, trainOutputs]
testData = [testInputs, testOutputs]

saveDataToFile("./model data/train.data", trainData)
saveDataToFile("./model data/test.data", testData)
