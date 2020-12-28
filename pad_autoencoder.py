import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

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

max_num_vias = None

def inputModel(inputLabel):
    layer_input = Input(shape=(max_num_vias,num_layers, ), name='layer-num-' + inputLabel) #Passed in as one hot 
    net_input = Input(shape=(max_num_vias, ), name='net-name-' + inputLabel)
    pos_input = Input(shape=(max_num_vias, num_pos_tokens, ), name='pos-' + inputLabel)
    size_input = Input(shape=(max_num_vias, num_size_tokens, ), name='size-' + inputLabel)

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


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, path="./model data/trainPads.data"):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data = loadData(path)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)      
        
        # Initialization
        encoder_input_data_pos= []
        encoder_input_data_size= []
        encoder_input_data_layer= []
        encoder_input_data_net= []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.data['id']

            # Store class
            y[i] = self.data['id']

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

        return inputData, y

    def elements(self, pcb, key):
        return [element[key] for element in pcb]
    
    def loadData(self, id):
        data = np.load(id)

        encoder_input_data_pos= []
        encoder_input_data_size= []
        encoder_input_data_layer= []
        encoder_input_data_net= []

        #Fill arrays
        for pcb in data:
            encoder_input_data_pos.append(self.elements(pcb,'pos'))
            encoder_input_data_size.append(self.elements(pcb,'size'))
            encoder_input_data_layer.append(self.elements(pcb,'layer'))
            encoder_input_data_net.append(self.elements(pcb,'net'))
        
        return encoder_input_data_pos, encoder_input_data_size, encoder_input_data_layer, encoder_input_data_net