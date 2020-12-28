# import tensorflow as tf
from scraperTools import seqGetDataFromFile


# arrayType = tf.constant



#Data properties
num_layers = 3 #1, 2, 11 no 4 layer boards

#Pad data


inputData = {
    "layer-num-encoder": encoder_input_data_layer,
    "net-name-encoder": encoder_input_data_net,
    "pos-encoder": encoder_input_data_pos,
    "size-encoder": encoder_input_data_size,

    "layer-num-decoder": encoder_input_data_layer,
    "net-name-decoder": encoder_input_data_net,
    "pos-decoder": encoder_input_data_pos,
    "size-decoder": encoder_input_data_size
}


