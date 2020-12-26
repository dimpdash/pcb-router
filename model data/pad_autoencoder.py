import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

Model = keras.models.Model
Input = keras.layers.Input
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Embedding = keras.layers.Embedding
Concatenate = keras.layers.Concatenate
Reshape = keras.layers.Reshape
Masking = keras.layers.Masking

#Params
latent_dim = 10
num_net_embedded_dim = 10
num_net_tokens = 100 #TODO find max number of net names in dataset
num_layers = 32
num_pos_tokens = 2
num_size_tokens = 2

num_decoder_tokens = num_net_embedded_dim + num_layers + num_pos_tokens + num_size_tokens

def inputModel(inputLabel):
    layer_input = Input(shape=(None,num_layers, ), name='layer-num-' + inputLabel) #Passed in as one hot 
    net_input = Input(shape=(None, ), name='net-name-' + inputLabel)
    pos_input = Input(shape=(None, num_pos_tokens, ), name='pos-' + inputLabel)
    size_input = Input(shape=(None, num_size_tokens, ), name='size-' + inputLabel)

    # layer_input = Masking(input_shape=(None,num_layers, ), name='layer-num-' + inputLabel) #Passed in as one hot 
    # net_input = Input(shape=(None, ), name='net-name-' + inputLabel)
    # pos_input = Masking(input_shape=(None, num_pos_tokens, ), name='pos-' + inputLabel)
    # size_input = Masking(input_shape=(None, num_size_tokens, ), name='size-' + inputLabel)

    layer_input_masked = Masking()(layer_input)
    pos_input_masked = Masking()(pos_input) 
    size_input_masked = Masking()(size_input)
    
    net_embedded = Embedding(num_net_tokens,num_net_embedded_dim, mask_zero=True)(net_input)
    print([layer_input_masked, size_input, pos_input, net_embedded])
    conc = Concatenate(name=inputLabel + "-inputs")([layer_input_masked, size_input_masked, pos_input_masked, net_embedded])
    # conc = Concatenate(name=inputLabel + "-inputs")([layer_input, size_input, pos_input, net_embedded])   

    return [layer_input, size_input, pos_input,net_input], conc

def autoencoder(inputs):
    #Encoder
    pad_encoder_input = inputs[0]
    pad_encoder = LSTM(latent_dim, return_state=True, name="pad-encoder")
    pad_encoder_outputs, state_h, state_c = pad_encoder(pad_encoder_input)
    pad_encoder_states = [state_h, state_c]
    
    #Decoder
    pad_decoder_input = inputs[1]
    pad_decoder = LSTM(latent_dim, return_sequences=True, return_state=True, name="pad-decoder")
    pad_decoder_outputs, _, _ = pad_decoder(pad_decoder_input, initial_state=pad_encoder_states)
    pad_decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(pad_decoder_outputs)
    return pad_decoder_outputs