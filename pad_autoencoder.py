from weakref import WeakKeyDictionary
import numpy as np
from numpy.core.defchararray import array
import tensorflow as tf
from tensorflow import keras
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

p = 100
hilbert_curve = HilbertCurve(p, 2)

Model = keras.models.Model
Input = keras.layers.Input
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Embedding = keras.layers.Embedding
Concatenate = keras.layers.Concatenate
Reshape = keras.layers.Reshape
Masking = keras.layers.Masking
pad_sequences = keras.preprocessing.sequence.pad_sequences
# TextVectorization = keras.layers.experimental.preprocessing.TextVectorization
Sequence = keras.utils.Sequence
Tokenizer = keras.preprocessing.text.Tokenizer

#Params
latent_dim = 60
num_input_nets_dim = 64 # cannot include all nets as embedding layer can only handle 1000
num_net_embedded_dim = 10
num_net_tokens = 100 #TODO find max number of net names in dataset
max_net_len = 10 #TODO find max len of nets
num_layers = 2
num_pos_tokens = 2
num_size_tokens = 2
num_EOF_tokens = 2
num_net_tokens = 0

num_decoder_tokens = num_layers + num_net_tokens + num_pos_tokens + num_size_tokens + num_EOF_tokens
num_encoder_tokens = num_layers + num_net_tokens + num_pos_tokens + num_size_tokens + num_EOF_tokens
num_output_tokens = num_layers + num_net_tokens + num_pos_tokens + num_size_tokens + num_EOF_tokens

max_num_vias = None

encoder_input_name = 'encoder-input'
decoder_input_name = 'decoder-input'
encoder_input_net_name = 'encoder-input-net'
decoder_input_net_name = 'decoder-input-net'
encoder_input_mask_name = 'encoder-input-mask'
decoder_input_mask_name = 'decoder-input-mask'

UNCOMMON_NAME = 'UNCOMMON'

def simpleInputs():
    encoder_input = Input(shape=(None,num_decoder_tokens,), name=encoder_input_name)
    decoder_input = Input(shape=(None,num_decoder_tokens,), name=decoder_input_name)
    encoder_input_net = Input(shape=(None,), name=encoder_input_net_name)
    decoder_input_net = Input(shape=(None,), name=decoder_input_net_name)
    
    encoder_input_mask = Input(shape=(None,), name=encoder_input_mask_name, dtype=bool)
    decoder_input_mask = Input(shape=(None,), name=decoder_input_mask_name, dtype=bool)
    return encoder_input, decoder_input, encoder_input_net, decoder_input_net


def inputModel(inputLabel):
    layer_input = Input(shape=(None, num_layers, ), name='layer-num-' + inputLabel) #Passed in as one hot 
    net_input = Input(shape=(1, ), name='net-name-' + inputLabel, dtype=tf.string)
    pos_input = Input(shape=(None, num_pos_tokens, ), name='pos-' + inputLabel)
    size_input = Input(shape=(None, num_size_tokens, ), name='size-' + inputLabel)

    # layer_input_masked = Masking()(layer_input)
    # pos_input_masked = Masking()(pos_input) 
    # size_input_masked = Masking()(size_input)
    
    net_embedded = Embedding(100,num_net_embedded_dim)(net_vectorized)
    
    # conc = Concatenate(name=inputLabel + "-inputs")([layer_input_masked, size_input_masked, pos_input_masked, net_embedded])
    conc = Concatenate(name=inputLabel + "-inputs")([layer_input, size_input, pos_input, net_embedded])   

    # conc_masked = Masking()(conc)
    return [layer_input, size_input, pos_input,net_input], conc

def encoder(net_embedding, numeric_encoder_input, pad_encoder_input_net, pad_encoder):
    encoder_net_embedded = net_embedding(pad_encoder_input_net)
    pad_encoder_input = Concatenate(axis=-1)([encoder_net_embedded, numeric_encoder_input])
    pad_encoder_input._keras_mask = encoder_net_embedded._keras_mask

    pad_encoder_outputs, state_h, state_c = pad_encoder(pad_encoder_input)
    pad_encoder_states = [state_h, state_c]

    return pad_encoder_states

def decoder(net_embedding_decoder, numeric_decoder_input, pad_decoder_input_net, LSTM_decoder, Dense_decoder):
    decoder_net_embedded = net_embedding_decoder(pad_decoder_input_net)
    pad_decoder_input = Concatenate(axis=-1)([numeric_decoder_input,decoder_net_embedded])
    pad_decoder_input._keras_mask = decoder_net_embedded._keras_mask

    pad_decoder_outputs, _, _ = pad_decoder(pad_decoder_input, initial_state=pad_encoder_states)
    pad_decoder_outputs = Dense_decoder(pad_decoder_outputs)
    return pad_decoder_outputs

###### Training Model #######
def autoencoder(inputs):
    net_embedding = Embedding(num_input_nets_dim + 1, num_net_embedded_dim,mask_zero=True)
    net_embedding_decoder = Embedding(num_input_nets_dim + 1, num_net_embedded_dim,mask_zero=True)

    #Encoder
    numeric_encoder_input = inputs[0]
    pad_encoder_input_net = inputs[2]

    pad_encoder = LSTM_encoder(latent_dim, return_state=True, name="pad-encoder")
    
    pad_encoder_states = encoder(net_embedding, numeric_encoder_input, pad_encoder_input_net, pad_encoder)
    
    #Decoder
    numeric_decoder_input = inputs[1]
    pad_decoder_input_net = inputs[3]

    pad_decoder = LSTM(latent_dim, return_sequences=True, return_state=True, name="pad-decoder")
    dense_decoder =  Dense(num_decoder_tokens, activation='relu')

    pad_decoder_outputs = decoder(net_embedding_decoder, numeric_decoder_input, pad_decoder_input_net, pad_decoder, dense_decoder)

    return pad_decoder_outputs

def padAutoencoder():
    encoder_input_list, encoder_input_conc = inputModel("encoder")
    decoder_input_list, decoder_input_conc = inputModel("decoder")

    pad_out = autoencoder([encoder_input_conc, decoder_input_conc])
    return Model(encoder_input_list + decoder_input_list, pad_out)

############ Inference Model ############


# Define sampling models
# Restore the model and construct the encoder and decoder.
         
def inferenceModelEncoder(trainingModel):
    print(trainingModel.summary())
    numeric_encoder_input = trainingModel.inputs[0]  # input_1
    pad_encoder_input_net = trainingModel.inputs[2]
    pad_encoder = trainingModel.get_layer(name='pad-encoder')
    net_embedding = trainingModel.get_layer(name='embedding')

    encoder_states = encoder(net_embedding, numeric_encoder_input, pad_encoder_input_net, pad_encoder)
    encoder_model = keras.Model([numeric_encoder_input, pad_encoder_input_net], encoder_states)
    return encoder_model
    # decoder_inputs = trainingModel.input[1]  # input_2
    # decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
    # decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
    # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    # decoder_lstm = trainingModel.layers[3]
    # decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    #     decoder_inputs, initial_state=decoder_states_inputs
    # )
    # decoder_states = [state_h_dec, state_c_dec]
    # decoder_dense = trainingModel.layers[4]
    # decoder_outputs = decoder_dense(decoder_outputs)
    # decoder_model = keras.Model(
    #     [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    # )
    

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

def replaceTexts(net):
    return net.replace(' ','_')

class MyTokenizer():
    # Tokenizer class that has net names beyond the word limit set as index 1 rather than index 0. leaving index 0 for masking
    def __init__(self, filters='', num_words=None):
        self.num_words = num_words
        self.tokenizer = Tokenizer(filters=filters, num_words=num_words)
        # self.uncommon_name = UNCOMMON_NAME
        # self.tokenizer.fit_on_texts([self.uncommon_name]) # reserve index 1 as uncommon name leaving 0 as a mask

    def texts_to_sequences(self, text):
        seqs = self.tokenizer.texts_to_sequences(text)

        for i, seq in enumerate(seqs):
            if not seq:
                seqs[i] = [self.num_words]

        return seqs
    def fit_on_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)
    def sequences_to_texts(self, seqs):
        return self.tokenizer.sequences_to_texts(seqs)

class loadable():
    def __init__(self, folderPath):
         self.folderPath = folderPath
    
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
        net = list(map(replaceTexts, net)) 
        
        return {"pos": pos, "size": size, "layer": layer, "net": net}
    

    def loadAllData(self):
        data = {}
        for id in self.list_IDs:
            data[id] = self.loadData(id)
        return data

class DataPipeline(loadable):
    def __init__(self, ids, folderPath, batchSize=32, validation_split=0.1):
        self.folderPath = folderPath
        self.ids = ids
        cutoffIndex = int(len(ids)*validation_split)
        trainIds = ids[cutoffIndex:]
        valIds = ids[:cutoffIndex]

        self.tokenizer = MyTokenizer(filters='',num_words=num_input_nets_dim-1)
        self.addWordsToTokenizer()

        self.preprocessDataStats()

        self.trainData = DataGenerator(trainIds, stats=self.stats, batch_size = batchSize, tokenizer=self.tokenizer, folderPath=self.folderPath)
        self.valData = DataGenerator(valIds, stats=self.stats, batch_size = batchSize, tokenizer=self.tokenizer, folderPath=self.folderPath)

    def dataGenerators(self):
        return self.trainData, self.valData

    def addWordsToTokenizer(self):
        for id in self.ids:
            pcb = self.loadData(id)
            net = pcb["net"]
            self.tokenizer.fit_on_texts(net)

    def preprocessDataStats(self):
        posxStats = Stats()
        posyStats = Stats()
        sizexStats = Stats()
        sizeyStats = Stats()

        for id in self.ids:
            data = self.loadData(id)
            pos = np.transpose(data["pos"]) 
            size = np.transpose(data["size"])
            posxStats.addDataPoints(pos[0])
            posyStats.addDataPoints(pos[1])
            sizexStats.addDataPoints(size[0])
            sizeyStats.addDataPoints(size[1])

        self.stats = [posxStats, posyStats, sizexStats, sizeyStats]

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

def renameUncommonNets(net, commonNets):
    if net not in commonNets:
        return UNCOMMON_NAME
    else:
        return net

def normalizeAndStandardize(x, stats):
    x = standardize(x, stats.mean(), stats.std())
    maxVal, minVal = standardize((stats.max, stats.min), stats.mean(), stats.std()) #rescale mean and std
    return normalize(x, minVal, maxVal)

def normalize(x, min, max):
    return (x - min)/(max-min)

def standardize(x, mean, std):
    return (x - mean)/std




class DataGenerator(Sequence, loadable):
    'Generates data for Keras'
    def __init__(self, list_IDs, stats, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, folderPath="./model data/pads/", tokenizer=MyTokenizer(filters='')):
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
        self.posxStats, self.posyStats, self.sizexStats, self.sizeyStats = stats
        self.minPos = min(self.posxStats.min, self.posyStats.min)
        self.posxi = 4
        self.posyi = 5
    
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

        #find max number of pads in batch
        max_pads = 0
        for key in pcbs:
            if max_pads < len(pcbs[key]["pos"]):
                max_pads = len(pcbs[key]["pos"])

        encoderInput = []
        decoderInput = []
        encoderInputNet = []
        decoderInputNet = []
        y = []
        encoderInputMask = np.zeros((self.batch_size, max_pads), dtype=bool)
        decoderInputMask = np.zeros((self.batch_size, max_pads), dtype=bool)

        # Store sample
        for i, key in enumerate(pcbs):
            pos = np.transpose(pcbs[key]["pos"]) 
            size = np.transpose(pcbs[key]["size"])
            posx = normalizeAndStandardize(pos[0], self.posxStats)
            posy = normalizeAndStandardize(pos[1], self.posyStats)
            sizex = normalizeAndStandardize(size[0], self.sizexStats)
            sizey = normalizeAndStandardize(size[1], self.sizeyStats)

            layerCategorical = pcbs[key]["layer"]
            layer = np.zeros((len(layerCategorical), 2))
            for j, x in enumerate(layerCategorical):
                if x == 0 or x == 1:
                    layer[j][x] = 1 
            
            netNames = pcbs[key]["net"]
            net = self.net_tokenizer.texts_to_sequences(netNames)
            net = np.array(net).flatten()

            sof = np.zeros(len(layer))
            eof = sof
            pcbFormatted = np.dstack((sof, eof, layer[:,0], layer[:,1], posx, posy, sizex, sizey))[0]
            pcbFormatted = self.orderPoints(pcbFormatted, np.transpose(pos))
            startTag = np.zeros((1,pcbFormatted.shape[1]))
            startTag[0][0] = 1
            endTag = np.zeros((1,pcbFormatted.shape[1]))
            endTag[0][1] = 1
            padding = np.zeros((max_pads-pcbFormatted.shape[0], pcbFormatted.shape[1]))
            paddingNet = np.zeros(max_pads-net.shape[0])

            encoderInput.append(np.concatenate((startTag, pcbFormatted, endTag, padding), axis=0))
            decoderInput.append(np.concatenate((startTag, pcbFormatted, padding)))
            y.append(np.concatenate((pcbFormatted, endTag, padding)))
            encoderInputNet.append(np.concatenate(([1],net,[1],paddingNet)))
            decoderInputNet.append(np.concatenate(([1],net,paddingNet)))
            # encoderInputMask[i][:pcbFormatted.shape[0] + 2] = True # add 2 for end and start tage
            # decoderInputMask[i][:pcbFormatted.shape[0]  + 1] = True # add one for start tag
            

        arrayType = np.array
        encoderInput = arrayType(encoderInput, dtype=np.float64)
        decoderInput = arrayType(decoderInput,  dtype=np.float64)
        y = arrayType(y,  dtype=np.float64)
        encoderInputNet = arrayType(encoderInputNet, dtype=np.float64)
        decoderInputNet = arrayType(decoderInputNet, dtype=np.float64)

        # layer = np.array([pcbs[key]["layer"] for key in pcbs])
        # net = np.array([pcbs[key]["net"] for key in pcbs])

        #Pad data
        # layer = pad_sequences(layer, padding='post')
        # size1 = pad_sequences(size1, padding='post')
        # size2 = pad_sequences(size2, padding='post')
        # net = pad_sequences(net, padding='post') # '' is the padding token for the textVectorization layer
        # pos1 = pad_sequences(pos1, padding='post')
        # pos2 = pad_sequences(pos2, padding='post')
        
        X = {encoder_input_name: encoderInput,
            decoder_input_name: decoderInput,
            encoder_input_net_name: encoderInputNet,
            decoder_input_net_name: decoderInputNet
        }
        # print('X')
        # print(X)
        # print('Y')
        # print(y)
        self.X = X
        self.y = y
        return X, y
    
    def orderPoints(self, pcb, points):
        distances = np.array(hilbert_curve.distances_from_points(((points + abs(self.minPos))*100).astype('int')))
        inds = distances.argsort()
        t = pcb[inds]
        return t


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



