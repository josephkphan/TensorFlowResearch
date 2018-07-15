#!/usr/local/bin/python3
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import imdb

# Download the dataset
imdb.data_dir = "./data/"
imdb.maybe_download_and_extract()

# Load the dataset
# Should Ouput:
#   Train-set size:  25000
#   Test-set size:   25000
x_train_text, y_train = imdb.load_data(train=True)
x_test_text, y_test = imdb.load_data(train=False)
print("Train-set size: ", len(x_train_text))
print("Test-set size:  ", len(x_test_text))


# Combine into one data-set for some uses below.
data_text = x_train_text + x_test_text

# Print an example from the training-set to see that the data looks correct.
print(x_train_text[1])

#  The ground truth value for the review (between 0-1)
print(y_train[1])


# Keras has already built a tokenizer for building a vocabulary 
num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data_text)


if  num_words  is None:
    num_words = len(tokenizer.word_index)

tokenizer.word_index

x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
x_train_text[1]
np.array(x_train_tokens[1])

print(np.array(x_train_tokens[1]))