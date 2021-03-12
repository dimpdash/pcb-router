#raw data --> pcb files
from pcbExtractor import main as rawToPCB
print('Beginning conversion from raw to PCB')
rawToPCB()

# pcb files --> pcb files json
from pcbExtractorToJSON import main as pcbToPcbJson
print('Beginning conversion from pcb to pcb with JSON')
pcbToPcbJson()

# pcb files json --> model data (pads and vias)
from createTrainingAndTestDataScript import main as pcbJsonToModelData
print('Beginng conversion from pcb with JSON to model data')
pcbJsonToModelData()

# model data ids