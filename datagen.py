from pad_autoencoder import DataGenerator, autoencoder, inputModel, simpleInputs
import numpy as np
from tensorflow import keras
from scraperTools import getDataFromFile
import random

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
cutoffIndex = int(len(ids)*validation_split)
opt = keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1)

encoder_input, decoder_input = simpleInputs()

pad_out = autoencoder([encoder_input, decoder_input])
model = Model([encoder_input, decoder_input], pad_out)

tokenizer = Tokenizer(filters='')

trainIds = ids[cutoffIndex:]
valIds = ids[:cutoffIndex]
trainData = DataGenerator(trainIds, batch_size = batchSize, tokenizer=tokenizer)
valData = DataGenerator(valIds, batch_size = batchSize, tokenizer=tokenizer)

model.compile(optimizer=opt, loss='mse', run_eagerly=True)
model.fit(x=trainData, validation_data = valData, epochs = 100)
model.save('model')