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
pad_sequences = keras.preprocessing.sequence.pad_sequences
TextVectorization = keras.layers.experimental.preprocessing.TextVectorization
Sequence = keras.utils.Sequence

#Params
latent_dim = 10
num_net_embedded_dim = 10
num_net_tokens = 100 #TODO find max number of net names in dataset
max_net_len = 10 #TODO find max len of nets
num_layers = 3
num_pos_tokens = 2
num_size_tokens = 2

num_decoder_tokens = num_net_embedded_dim + num_layers + num_pos_tokens + num_size_tokens

max_num_vias = None

def inputModel(inputLabel):
    layer_input = Input(shape=(None, num_layers, ), name='layer-num-' + inputLabel) #Passed in as one hot 
    net_input = Input(shape=(1, ), name='net-name-' + inputLabel, dtype=tf.string)
    pos_input = Input(shape=(None, num_pos_tokens, ), name='pos-' + inputLabel)
    size_input = Input(shape=(None, num_size_tokens, ), name='size-' + inputLabel)

    # layer_input_masked = Masking()(layer_input)
    # pos_input_masked = Masking()(pos_input) 
    # size_input_masked = Masking()(size_input)
    
    net_vectorized = TextVectorization(output_mode='int')(net_input)
    net_embedded = Embedding(num_net_tokens,num_net_embedded_dim)(net_vectorized)
    
    # conc = Concatenate(name=inputLabel + "-inputs")([layer_input_masked, size_input_masked, pos_input_masked, net_embedded])
    conc = Concatenate(name=inputLabel + "-inputs")([layer_input, size_input, pos_input, net_embedded])   

    # conc_masked = Masking()(conc)
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

def padAutoencoder():
    encoder_input_list, encoder_input_conc = inputModel("encoder")
    decoder_input_list, decoder_input_conc = inputModel("decoder")

    pad_out = autoencoder([encoder_input_conc, decoder_input_conc])
    return Model(encoder_input_list + decoder_input_list, pad_out)

def addStartAndEndTags(lstlst, start, end):
    return [np.concatenate(([lst], lst, [end]), axis=None) for lst in lstlst]

def addEndTag(lstlst, end):
    return [np.concatenate((lst,[end]), axis=None) for lst in lstlst]

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, folderPath="./model data/pads/"):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.folderPath = folderPath
        self.data = self.loadAllData()
    

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
        X, y = self.data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            

    def data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)      
        
        # Initialization

        pcbs = {}
        for _, id in enumerate(list_IDs_temp):
            pcbs[id] = self.loadData(id)

        # batchSize = len(pcbs)

        # pos = np.zeros(batchSize)
        # size = np.zeros(batchSize)
        # layer = np.zeros(batchSize)
        # net = np.zeros(batchSize)

        # Store sample
        pos = [pcbs[key]["pos"] for key in pcbs]
        size = [pcbs[key]["size"] for key in pcbs]
        layer = [pcbs[key]["layer"] for key in pcbs]
        net = [pcbs[key]["net"] for key in pcbs]
        #Pad data
        layer = pad_sequences(layer, padding='post')
        size = pad_sequences(size, padding='post')
        net = pad_sequences(net, dtype=object, value='', padding='post') # '' is the padding token for the textVectorization layer
        pos = pad_sequences(pos, padding='post')
        
        netNew = [np.array(' '.join(netList)) for netList in net]

        # layer = np.array(layer)
        # size = np.array(size)
        # net = np.array(net)
        # pos = np.array(pos)

        netNameDecoderInput = [np.array( 'SOF ' + ' '.join(netList) + ' EOF') for netList in net]
        netNameDecoderOutput = [np.array( 'SOF ' + ' '.join(netList) + ' EOF') for netList in net]

        X = {
            "layer-num-encoder": layer,
            "net-name-encoder": netNew,
            "pos-encoder": pos,
            "size-encoder": size,

            "layer-num-decoder": addStartAndEndTags(layer,-1,-2),
            "net-name-decoder":  netNameDecoderInput, #addStartAndEndTags(net,'SOF', 'EOF'), #TODO check SOF and EOF aren't used
            "pos-decoder": addStartAndEndTags(pos,-1,-2),
            "size-decoder": addStartAndEndTags(size,-1,-2)
        }

        y = {  
             addEndTag(layer,-2),
                netNameDecoderOutput, #addEndTag(net, 'EOF'), #TODO check SOF and EOF aren't used
            "pos-decoder": addEndTag(pos,-2),
            "size-decoder": addEndTag(size,-2)
        }

        return X, y

    def elements(self, pcb, key):
        return [element[key] for element in pcb]
    
    def loadData(self, id):
        pcb = np.load(self.folderPath + id + ".npy", allow_pickle=True)

        pos= []
        size= []
        layer= []
        net= []

        #Fill arrays
        pos = self.elements(pcb,'pos')
        size = self.elements(pcb,'size')
        layer = self.elements(pcb,'layer')
        net = self.elements(pcb,'net')
        
        return {"pos": pos, "size": size, "layer": layer, "net": net}
    
    def loadAllData(self):
        data = {}
        for id in self.list_IDs:
            data[id] = self.loadData(id)
        return data