import numpy as np
import pandas as pd

from getData import getData
from keras import Sequential, Model
from keras.layers import LSTM, Dense, Embedding, Activation, Input, concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import adam

# Parameters to fine tune tokenization
parameters = {"vocab_size": 500,
              "max_len": 100,
              "padding": "post",
              "truncate": "post",
              }


def prepareData():
    """
    function to prepare padded sequences of titles concatenated with texts for training and testing
    the dates and subjects of the news have been excluded as they don't explain much the news being real of fake
    :return:
        trainPad: numpy array of padded sequences of titles followed by texts for training
        trainLabels: numpy array of training labels [0 for real news, 1 for fake news]
        testPad: numpy array of padded sequences of titles followed by texts for testing
        testLabels: numpy array of testing labels [0 for real news, 1 for fake news]
    """

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    [[trainData, trainLabels], [testData, testLabels]] = getData(trainSize=0.9)
    categories = list(trainData)

    # Tokenize title and text
    tt_tokenizer = Tokenizer(num_words=parameters["vocab_size"])
    tt_tokenizer.fit_on_texts(trainData['title'] + " : " +
                              trainData['text'])

    trainSeq = tt_tokenizer.texts_to_sequences(trainData['title'] + " : " +
                                               trainData['text'])
    trainPad = pad_sequences(trainSeq,
                             maxlen=parameters["max_len"],
                             padding=parameters["padding"],
                             truncating=parameters["truncate"])

    testSeq = tt_tokenizer.texts_to_sequences(testData['title'] + " : " +
                                              testData['text'])
    testPad = pad_sequences(testSeq,
                            maxlen=parameters["max_len"],
                            padding=parameters["padding"],
                            truncating=parameters["truncate"])

    return [[np.array(trainPad), np.array(trainLabels)], [np.array(testPad), np.array(testLabels)]]


def initModel():
    """
    Function to initiate model
    :return: compiled keras functional model
    """

    model = Sequential()
    model.add(Embedding(input_dim=parameters["vocab_size"],
                        input_length=parameters["max_len"],
                        output_dim=100))
    model.add(LSTM(64))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=adam(), loss=binary_crossentropy, metrics=['accuracy'])

    return model


if __name__ == "__main__":
    prepareData()
