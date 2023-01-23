"""
The objective of this code is to perform sentiment analysis and create a lstm model for this data
"""
import pickle
import re
from collections import Counter
import os
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import tensorflow
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping
from keras.layers import (LSTM, Bidirectional, Dense, Dropout, Embedding,Conv1D,MaxPooling1D,Flatten,
                          SpatialDropout1D)
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def csv_to_dataframe(file_name):
    """
    Read a csv file and create a dataframe

    Parameters:
    ----------
    file_name : csv file name

    Returns:
    -------
    df : dataframe for that csv file
    """
    
    df=pd.read_csv(file_name , encoding="utf-8",delimiter=',',quotechar='"')
    return df

# load embedding as a dict
def load_embedding(filename):
    """
    Load an embbeding layer based on a file provided

    Parameters:
    ----------
    filename : Text file containing words and their transformation into vectors
    Returns:
    -------
    embedding : A dictionary of words in the file provided with their related vectors
    
    """
    # load embedding into memory, skip first line
    file = open(filename,mode='r',encoding="utf-8")
    lines = file.readlines()
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    """
    Use a vocabulary of words wih embedding provided for finding each vector for word in the vocabulary

    Parameters:
    ----------
    embedding : embedding for all words 
    vocab : all words in the documents being proccessed

    Returns:
    -------
    weight_matrix : list of words in their vector format based on embedding from outside source
    """
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, embedding_dim))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix


if __name__ == "__main__":

    file_name='tweets_labeled.csv'
    df=csv_to_dataframe(file_name=file_name)
    df['tokenized_tweets']=df['cleaned_text'].apply(lambda x : nltk.word_tokenize(str(x)))
    le = LabelEncoder()
    df['sentiment_labeled'] = le.fit_transform(df['sentiment'])
    Y=pd.get_dummies(df['sentiment_labeled'])

    # Max number of words in each tweets.
    MAX_SEQUENCE_LENGTH = max_length = max([len(s) for s in df.tokenized_tweets])

    tokenizer = Tokenizer(split=' ') 
    tokenizer.fit_on_texts(df.tokenized_tweets)
    vocab = tokenizer.word_index
    vocab_size=len(vocab)+1
    print('Found %s unique tokens.' % len(vocab))

    X = tokenizer.texts_to_sequences(df.tokenized_tweets.values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)

    np.random.seed(1)
    tensorflow.random.set_seed(2)

    f1 = tfa.metrics.F1Score(36,'micro' or 'macro')
    embedding_dim = 50
    # load embedding from file
    raw_embedding = load_embedding('glove.6B.50d.txt')
    # get vectors in the right order
    embedding_matrix = get_weight_matrix(raw_embedding, vocab)

    embedding_layer = Embedding(vocab_size, 
                                embedding_dim, 
                                weights = [embedding_matrix], 
                                input_length = max_length, 
                                trainable = False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(176, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.summary())
    EPOCHS = 14


    # train model
    history = model.fit(X_train, Y_train, epochs = EPOCHS,verbose=2)

    accr = model.evaluate(X_test,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    model.save('model_final.h5')

    # plotting model performance
    acc = history.history['accuracy']
    loss = history.history['loss'] 

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label = 'Training acc')
    plt.title('Training accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label = 'Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()

    Y_train_pred=model.predict(X_train)
    Y_test_pred=model.predict(X_test)
