from utils import setUpLogger
logger = setUpLogger()

from tensorflow import keras
from pad_autoencoder import DataGenerator, DataPipeline, getIds, inferenceModelEncoder, inferenceModelDecoder
from pad_autoencoder import num_decoder_tokens, decoder_input_name, decoder_input_net_name, state_c_name, state_h_name
import pickle
import numpy as np

ids = getIds()

fileHandler = open('dataPipeline_metaData.data','rb')
dataPipeline = pickle.load(fileHandler)
generator = DataGenerator(ids, dataPipeline.stats, tokenizer=dataPipeline.tokenizer)
testData = generator.data_generation([ids[0]])
# print(testData)

version = 2
modelFileName = 'model-v' + str(version)

trainingModel = keras.models.load_model(modelFileName)
print(trainingModel.summary())
encoder_model = inferenceModelEncoder(trainingModel)
encoder_input = [testData[0]['encoder-input'], testData[0]['encoder-input-net']]
# print(encoder_input)
encoder_states = encoder_model.predict(encoder_input)


# Reverse-lookup token index to decode sequences back to
# something readable.
# reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
# reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    decoder_model = inferenceModelDecoder(trainingModel)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq_net = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 0] = 1.0
    target_seq_net[0,0] = 1

    decoder_inputs = {decoder_input_name: target_seq,
        decoder_input_net_name: target_seq_net,
        state_h_name: states_value[0],
        state_c_name: states_value[1]
    }

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens = decoder_model.predict(decoder_inputs)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence


output = decode_sequence(encoder_input)
print(output)