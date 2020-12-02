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

def parsePoints(pointText):
        coordList = pointText.split()
        xcoords = coordList[0::2]
        ycoords = coordList[1::2]
        return [pair for pair in zip(xcoords, ycoords)]

def createDictForAttributes(shape):
    shapeTypes = {'F', 'SOLIDREGION', 'R', 'A', 'LIB', 'HOLE', 'COPPERAREA', 'ARC', 'T', 'SVGNODE', 'DIMENSION', 'W', 'RECT', 'VIA', 'TRACK', 'TEXT', 'P', 'N', 'PG', 'E', 'PAD', 'PL', 'CIRCLE', 'J'}
    attributeData = None
    if "TRACK" in shape:
        attributes = shape["TRACK"].split("~")
        attributeData = dict()

        attributeData["width"] = attributes[0]
        attributeData["layer"] = attributes[1]
        attributeData["net"] = attributes[2]
        attributeData["points"] = parsePoints(attributes[3])
        attributeData["id"] = attributes[4]

    if "VIA" in shape:
        attributes = shape["VIA"].split("~")
        attributeData = dict()

        attributeData["loc"] = (attributes[0],attributes[1])
        attributeData["outerDiameter"] = attributes[2]
        attributeData["net"] = attributes[3]
        attributeData["innerDiameter"] = attributes[4]
        attributeData["id"] = attributes[5]

    if "HOLE" in shape:
        attributes = shape["HOLE"].split("~")
        attributeData = dict()

        attributeData["loc"] = (attributes[0],attributes[1])
        attributeData["diameter"] = attributes[2]
        attributeData["id"] = attributes[3]
        try:
            attributes["undetected"] = attributes[5:]
        except:
            pass
    
    if "PAD" in shape:
        attributes = shape["PAD"].split("~")
        attributeData = dict()

        attributeData["type"] = attributes[0]
        attributeData["loc"] = (attributes[1],attributes[2])
        attributeData["width"] = attributes[3]
        attributeData["height"] = attributes[4]
        attributeData["layer"] = attributes[5]

        if attributeData["type"] == "ELLIPSE":
            attributeData["net"] = attributes[6]
            # attributeData["points"] = parsePoints(attributes[9])
            attributeData["id"] = attributes[11]

        elif attributeData["type"] == "RECT":
            attributeData["net"] = attributes[7]
            # attributeData["points"] = parsePoints(attributes[9])
            attributeData["id"] = attributes[11]

        elif attributeData["type"] == "OVAL":
            attributeData["net"] = attributes[6]
            # attributeData["loci"] = parsePoints(attributes[9])
            attributeData["id"] = attributes[11]

        elif attributeData["type"] == "POLYGON":
            attributeData["net"] = attributes[6]
            # attributeData["points"] = parsePoints(attributes[9])
            attributeData["id"] = attributes[11]
    
    if "LIB" in shape:
        attributes = shape["LIB"].split("~")
        attributeData = dict()

        attributeData["loc"] = (attributes[0], attributes[1])
        attributeData["package"] = attributes[2].split("`")[1]
        attributeData["id"] = attributes[5]

    if attributeData == None:
        for key in shape.keys():
            if key not in shapeTypes:
                print(key," is not a known shape type")
                raise
        # attributeData = list(shape.values())[0]
    return attributeData

def createDictForShapes(shapeDatas):
    pcbParsed = []
    for shape in shapeDatas:
        [shapeType, rawShapeData] = shape.split("~", maxsplit=1)
        shapeData = createDictForAttributes({shapeType:rawShapeData})
        if shapeType == "LIB":
            subshapes = shape.split("#@$")[1:]
            if subshapes != ['']:
                tmp = createDictForShapes(subshapes)
                pcbParsed.append({shapeType:tmp})
        else:
            pcbParsed.append({shapeType:shapeData})
    return pcbParsed