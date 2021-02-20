from pad_autoencoder import DataGenerator, autoencoder, inputModel
import numpy as np
from tensorflow import keras
from scraperTools import getDataFromFile
import random

Input = keras.layers.Input
Model = keras.Model

plot_model = keras.utils.plot_model


myGen = DataGenerator(list_IDs = ["000a5fc5e50b4009bbb09756d16790a8", "00a4d405afd84d5c8246ea1bea68fad1", "00b0bbb4728f49a5b4c40422f96e39a0", "00cb564aa9724961a3001f32ebe10a19"], batch_size = 2)

print(myGen.__getitem__(0))