import json

# class PCB():
#     __init__(self,data):
#         shapeData = data.["shape"]
#         self.tracks = self.getTracks(shapeData)

# def parseShapeData

def getFeatures(shapeData):
    shapeTypes = set()
    for shape in shapeData:
        shapeType = shape.split("~")[0]
        if shapeType == "LIB":
            subshapes = shape.split("#@$")[1:]
            inlib =getFeatures(subshapes)
            shapeTypes = shapeTypes.union(inlib)
        shapeTypes.add(shapeType)

    return shapeTypes

def createDictForShapes(shapeDatas):
    pcbParsed = []
    for shape in shapeDatas:
        if len(shape.split("~", maxsplit=1)) != 2:
            print(shapeDatas)
        try:
            [shapeType, shapeData] = shape.split("~", maxsplit=1)
            if shapeType == "LIB":
                subshapes = shape.split("#@$")[1:]
                tmp = createDictForShapes(subshapes)
                pcbParsed.append({shapeType:tmp})
            else:
                pcbParsed.append({shapeType:shapeData})
        except:
            pass
    return pcbParsed

def createDictForAttributes(shape):
    if "Track" in shape:
        attributes = shape["Track"].split("~")
        attributeData = dict()

        attributeData["width"] = attributes[0]

        attributeData["layer"] = attributes[1]

        coordList = attributes[2].split()
        xcoords = coordList[0::2]
        ycoords = coordList[1::2]
        attributeData["points"] = [list(pair) for pair in zip(xcoords, ycoords)]

        attributeData["id"] = attributes[3]

        attributeData["unknown"] = attributes[4]
        try:
            attributeData["undetected"] = attributeData[5:]
        except:
            pass
    if "VIA" in shape:
        attributes = shape["Track"].split("~")
        attributeData = dict()

        attributeData["loc"] = (attributes[0],attributes[1])
        attributeData["outerDiameter"] = attributes[2]
        attributeData["name"] = attributes[3]
        attributeData["innerDiameter"] = attributes[4]
        attributeData["id"] = attributes[5]
        attributes["unknown"] = attributes[6]
        
        try:
            attributes["undetected"] = attributes[7:]
        except:
            pass
    if "HOLE" in shape:
        attributes = shape["Track"].split("~")
        attributeData = dict()

        attributeData["loc"] = (attributes[0],attributes[1])
        attributeData["diameter"] = attributes[2]
        attributeData["id"] = attributes[3]
        attributes["unknown"] = attributes[4]
        try:
            attributes["undetected"] = attributes[5:]
        except:
            pass
    
    if "PAD" in shape:
        attributes = shape["Track"].split("~")
        attributeData = dict()

        attributeData["type"] = attributes[0]
        attributeData["loc"] = (attributes[1],attributes[2])
            # if attributeData["type"] = "ELLIPSE"

        try:
            attributes["undetected"] = attributes[5:]
        except:
            pass


