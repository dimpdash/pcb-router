# import tensorflow as tf
from pad_autoencoder import autoencoder, DataGenerator
from scraperTools import getDataFromFile

ids = getDataFromFile("./model data/ids.data")

dataGen = DataGenerator(ids)
#Pad data



