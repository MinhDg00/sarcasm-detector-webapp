import json
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Flatten, Dense, GlobalMaxPooling1D, Bidirectional, Conv1D, Dropout, GRU, LSTM,  MaxPooling1D
import warnings 
import pickle
warnings.filterwarnings('ignore')

#import data and label
sarcasm = []
sentences = []
labels = []
urls = []

for line in open('Sarcasm_Headlines_Dataset_v2.json','r'):
	sarcasm.append(json.loads(line))

for item in sarcasm:
	sentences.append(item['headline'])
	labels.append(item['is_sarcastic'])
	urls.append(item['article_link'])

#Hyperparameter

vocab_size = 10000
embedding_dim = 64
max_length = 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = 'Unknown'

# Split Data into train and test set
train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size = 0.2, random_state = 1)

# Preprocessing data 
# take in data and encode
tokenizer = Tokenizer(num_words = vocab_size,oov_token = oov_tok)
tokenizer.fit_on_texts(train_sentences)
# return dict comprise of key values pair: word and its indices
word_index = tokenizer.word_index

# return a sequence of indices representing wordssentences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

# pad sequences to same length
train_padded = pad_sequences(train_sequences, maxlen = max_length, 
                               padding = padding_type, truncating = trunc_type)
test_padded = pad_sequences(test_sequences, maxlen = max_length,
                              padding = padding_type, truncating = trunc_type)
# set seed
np.random.seed(46)
tf.random.set_seed(46)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    Dropout(0.2),
    Bidirectional(LSTM(64, return_sequences = True)),
    Bidirectional(LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
num_epochs = 5

test_labels = np.array(test_labels)
train_labels = np.array(train_labels)
 
model.fit(train_padded, train_labels, epochs = num_epochs, 
                    validation_data = (test_padded, test_labels), verbose = 2)


model.save('model.h5')

