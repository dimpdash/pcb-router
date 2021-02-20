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
Tokenizer = keras.preprocessing.text.Tokenizer

#Params
latent_dim = 128
num_net_embedded_dim = 10
num_net_tokens = 100 #TODO find max number of net names in dataset
max_net_len = 10 #TODO find max len of nets
num_layers = 2
num_pos_tokens = 2
num_size_tokens = 2
num_EOF_tokens = 2
num_net_tokens = 1

num_decoder_tokens = num_layers + num_net_tokens + num_pos_tokens + num_size_tokens + num_EOF_tokens

max_num_vias = None

def simpleInputs():
    encoder_input = Input(shape=(None,num_decoder_tokens,), name="encoder-input")
    decoder_input = Input(shape=(None,num_decoder_tokens,), name="decoder-input")
    return encoder_input, decoder_input

def inputModel(inputLabel):
    layer_input = Input(shape=(None, num_layers, ), name='layer-num-' + inputLabel) #Passed in as one hot 
    net_input = Input(shape=(1, ), name='net-name-' + inputLabel, dtype=tf.string)
    pos_input = Input(shape=(None, num_pos_tokens, ), name='pos-' + inputLabel)
    size_input = Input(shape=(None, num_size_tokens, ), name='size-' + inputLabel)

    # layer_input_masked = Masking()(layer_input)
    # pos_input_masked = Masking()(pos_input) 
    # size_input_masked = Masking()(size_input)
    
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

# def addStartAndEndTags(batch):
#     batchExtend = np.zeros((batch.shape[0], batch.shape[1] + 2, batch.shape[2] + 2))
#     for i, singBatch in enumerate(batch):
#         batchExtend[i][1:-1, 2:] = singBatch
#         batchExtend[i][0, 0] = 1
#         batchExtend[i][-1, 1] = 1
#     return batchExtend

# def addEndTag(batch):
#     batchExtend = np.zeros((batch.shape[0], batch.shape[1] + 2, batch.shape[2] + 1))
#     for i, singBatch in enumerate(batch):
#         batchExtend[i][:-1, 1:] = singBatch
#         batchExtend[i][-1, 0] = 1
#     return batchExtend

def addStartAndEndTags(batch):
    start = np.zeros(num_decoder_tokens)
    end = np.zeros(num_decoder_tokens)
    start[0] = 1
    end[1] = 1
    for i, singBatch in enumerate(batch):
        batch[i] = np.concatenate((start,batch[i],end))
    return batch

def addEndTag(batch):
    end = np.zeros(num_decoder_tokens)
    end[1] = 1
    for i, singBatch in enumerate(batch):
        batch[i] = np.concatenate((batch[i],end))
    return batch

def addStartAndEndRows(batch):
    # batchExtend = np.zeros((batch.shape[0], batch.shape[1] + 2, batch.shape[2] + 2))
    batchExtend = []
    for i, singBatch in enumerate(batch):
        # batchExtend[i][1:-1, 2:] = singBatch
        pcb = []
        for j in singBatch:
            for k in singBatch[j]:
                pcb = np.concatenate((np.zeros(2), singBatch[j][k]))
        batchExtend.append(pcb)
    return batchExtend

def addEmptyStringName(name):
    if name == '':
        return 'Unnamed'
    else:
        return name.replace(' ', '_')

def normalizeAndStandardize(x, stats):
    x = standardize(x, stats.mean(), stats.std())
    max, min = standardize((stats.max, stats.min), stats.mean(), stats.std()) #rescale mean and std
    return normalize(x, min, max)

def normalize(x, min, max):
    return (x - min)/(max-min)

def standardize(x, mean, std):
    return (x - mean)/std


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, folderPath="./model data/pads/", tokenizer=Tokenizer(filters='')):
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
        self.net_tokenizer = tokenizer
        self.preprocessDataStats()

    def preprocessDataStats(self):
        self.posxStats = Stats()
        self.posyStats = Stats()
        self.sizexStats = Stats()
        self.sizeyStats = Stats()

        for id in self.list_IDs:
            data = self.loadData(id)
            pos = np.transpose(data["pos"]) 
            size = np.transpose(data["size"])
            self.posxStats.addDataPoints(pos[0])
            self.posyStats.addDataPoints(pos[1])
            self.sizexStats.addDataPoints(size[0])
            self.sizeyStats.addDataPoints(size[1])
    
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

        encoderInput = []
        decoderInput = []
        y = []
        
        max_pads = 0
        for key in pcbs:
            if max_pads < len(pcbs[key]["pos"]):
                max_pads = len(pcbs[key]["pos"])

        # Store sample
        for key in pcbs:
            pos = np.transpose(pcbs[key]["pos"]) 
            size = np.transpose(pcbs[key]["size"])
            posx = normalizeAndStandardize(pos[0], self.posxStats)
            posy = normalizeAndStandardize(pos[1], self.posyStats)
            sizex = normalizeAndStandardize(size[0], self.sizexStats)
            sizey = normalizeAndStandardize(size[1], self.sizeyStats)

            layerCategorical = pcbs[key]["layer"]
            layer = np.zeros((len(layerCategorical), 2))
            for i, x in enumerate(layerCategorical):
                if x == 0 or x == 1:
                    layer[i][x] = 1 
            
            netNames = pcbs[key]["net"]
            netNames = list(map(addEmptyStringName, netNames))
            self.net_tokenizer.fit_on_texts(netNames)
            net = self.net_tokenizer.texts_to_sequences(netNames)
            net = np.array(net).flatten()

            sof = np.zeros(len(layer))
            eof = sof
            pcbFormatted = np.dstack((sof, eof, layer[:,0], layer[:,1], posx, posy, sizex, sizey, net))[0]
            startTag = np.zeros((1,pcbFormatted.shape[1]))
            startTag[0][0] = 1
            endTag = np.zeros((1,pcbFormatted.shape[1]))
            endTag[0][1] = 1
            padding = np.zeros((max_pads-pcbFormatted.shape[0], pcbFormatted.shape[1]))

            encoderInput.append(np.concatenate((startTag, pcbFormatted, endTag, padding), axis=0))
            decoderInput.append(np.concatenate((startTag, pcbFormatted, padding)))
            y.append(np.concatenate((pcbFormatted, endTag, padding)))

        arrayType = np.array
        encoderInput = arrayType(encoderInput, dtype=np.float64)
        decoderInput = arrayType(decoderInput,  dtype=np.float64)
        y = arrayType(y,  dtype=np.float64)

        # layer = np.array([pcbs[key]["layer"] for key in pcbs])
        # net = np.array([pcbs[key]["net"] for key in pcbs])

        #Pad data
        # layer = pad_sequences(layer, padding='post')
        # size1 = pad_sequences(size1, padding='post')
        # size2 = pad_sequences(size2, padding='post')
        # net = pad_sequences(net, padding='post') # '' is the padding token for the textVectorization layer
        # pos1 = pad_sequences(pos1, padding='post')
        # pos2 = pad_sequences(pos2, padding='post')
        
        X = {"encoder-input": encoderInput,
            "decoder-input": decoderInput
        }
        # print('X')
        # print(X)
        # print('Y')
        # print(y)
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

class Stats():
    def __init__(self):
        self.resetCount()
    
    def addDataPoints(self, xs):
        for x in xs:
            self.addDataPoint(x)

    def addDataPoint(self,x):
        self.n = self.n + 1 
        self.S1 = self.S1 + x
        self.S2 = self.S2 + x**2
        if self.min > x:
            self.min = x
        if self.max < x:
            self.max = x
    
    def mean(self):
        return self.S1/self.n
    
    def std(self):
        return np.sqrt(self.S2/self.n - (self.S1/self.n)**2)
    
    def resetCount(self):
        self.S1 = 0
        self.S2 = 0
        self.n = 0
        self.min = np.inf
        self.max = 0


