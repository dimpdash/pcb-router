from utils import setUpLogger
logger = setUpLogger()

from pad_autoencoder import DataPipeline, autoencoder, inputModel, simpleInputs
import numpy as np
from tensorflow import keras
from scraperTools import getDataFromFile
import random





##### Initalise Model ############# 
modelFileName = 'model-v'
Input = keras.layers.Input
Model = keras.Model
Tokenizer = keras.preprocessing.text.Tokenizer

ids = getDataFromFile("./model data/ids.data")

blackListedPCBs = []

for id in ids:
    pcb = np.load("./model data/pads/" + id + ".npy", allow_pickle=True)
    if len(pcb) == 0:
        blackListedPCBs.append(id)

print(blackListedPCBs)

for blackListed in blackListedPCBs:
    ids.remove(blackListed)

random.shuffle(ids)

batchSize = 32
validation_split = 0.1
folderPath = "./model data/pads/"
cutoffIndex = int(len(ids)*validation_split)
opt = keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)

inputs = simpleInputs()

pad_out = autoencoder(inputs)
model = Model(inputs, pad_out)
dataPipeline = DataPipeline(ids, folderPath, batchSize=batchSize, validation_split=validation_split)
trainData, valData = dataPipeline.dataGenerators()



# keras.utils.plot_model(model, "model.png", show_shapes=True)

lower = 2 #set to -1 to start new model 
upper = 10
if lower == -1:
    model.compile(optimizer=opt, loss='mse')
else: 
    model = keras.models.load_model(modelFileName + str(lower))

for i in range(lower + 1, upper):
    logger.info('Beginning iteration %d', i)
    model.fit(x=trainData, validation_data = valData, epochs = 10)
    model.save(modelFileName + str(i))
    logger.info('Finished iteration %d', i)