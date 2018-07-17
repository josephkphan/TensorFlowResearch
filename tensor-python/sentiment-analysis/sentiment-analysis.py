#!/usr/local/bin/python3
import argparse
import imdb
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datetime import datetime
from pprint import pprint, pformat
from scipy.spatial.distance import cdist
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# Globals
EMBEDDING_SIZE = 8
EXAMPLE_INDEX = 0       # Used for debug logs
NUM_WORDS = 10000       # maximum amount of words for vocabulary (by popularity)
PAD  =  'pre'           # Pad sequences

imdb.data_dir = "/tmp/sentiment-analysis"
imdb.data_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--debug', type=str, help='(true | false)', required=False, default="false" )
args = parser.parse_args()


# Setting up logger
if "true" in args.debug:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s ',filename='sa.log', filemode='w', datefmt='%H:%M:%S',level=logging.DEBUG)
else: 
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s ', datefmt='%H:%M:%S',level=logging.INFO)


# ------------------- RAW TEXT ------------------- #
logging.info("STARTING PHASE - RAW TEXT") 

# Load the datasets
imdb.maybe_download_and_extract()
x_train_text, y_train = imdb.load_data(train=True)
x_test_text, y_test = imdb.load_data(train=False)
logging.debug("Size - Train-set size: %s" % len(x_train_text) )    # Should Output Train-set size:  25000
logging.debug("Size - Test-set size:  %s" % len(x_test_text) )     # Should Output Train-set size:  25000
data_text = x_train_text + x_test_text                      # Combine into one data-set for some uses below.
logging.info("Finished loading up datasets") 


# Example from the training-set to see if the data looks correct.
logging.debug("Example - Training Entry #%s Text: %s" % (EXAMPLE_INDEX, x_train_text[EXAMPLE_INDEX])) # Example Review text sequence
logging.debug("Example - Training Entry #%s Value: %s" % (EXAMPLE_INDEX, y_train[EXAMPLE_INDEX]))     # The ground truth value for the review (between 0-1)


# ------------------- TOKENIZER ------------------- #
logging.info("STARTING PHASE - TOKENIZER") 

# Keras has already built a tokenizer for building a vocabulary for mapping words to integer tokens
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(data_text) 
logging.info("Finished creating tokenizer") 


# If NUM_WORDS was not set, set it to the number to unique words found (vocabulary size)
if  NUM_WORDS  is None:
    NUM_WORDS = len(tokenizer.word_index)

logging.debug(tokenizer.word_index)    # This is the vocabulary built
   
# Convert all of the sequences to tokens
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
x_test_tokens  = tokenizer.texts_to_sequences(x_test_text)
logging.info("Finished converting texts to tokens") 
logging.debug("Training Entry #%s Tokens:\n %s" % (EXAMPLE_INDEX , np.array(x_train_tokens[EXAMPLE_INDEX]))) 


# Gather Statistics of Data to determine Pad Size
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens] # Count number of words or tokens in every sequence
num_tokens = np.array(num_tokens)
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
logging.info("Finished calculating token statistics") 
logging.debug("Token Statistics - Mean: %s" % np.mean(num_tokens))
logging.debug("Token Statistics - Max: %s" % np.max(num_tokens))
logging.debug("Token Statistics - Token Threshold Caluclated: %s" % str(max_tokens))
logging.debug("Token Statistics - Sequence Coverage percentage: %s" % str((np.sum(num_tokens < max_tokens) / len(num_tokens))))

# Pad all data sequences
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens, padding=PAD, truncating=PAD)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens, padding=PAD, truncating=PAD)
logging.info("Finished Sequence Padding ") 
logging.debug("Padding - Training Entry #%s:\n %s" % (EXAMPLE_INDEX, str(x_train_pad[EXAMPLE_INDEX])))


idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))

def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]
    
    # Concatenate all words.
    text = " ".join(words)

    return text

logging.debug("Token to strings Entry #%s:  %s" % (EXAMPLE_INDEX, tokens_to_string(x_train_tokens[EXAMPLE_INDEX])))


# ------------------- EMBEDDING ------------------- #
logging.info("STARTING PHASE - EMBEDDING") 

# This is necessary because the integer-tokens may take on values between 0 and 10000 for a vocabulary 
# of 10000 words. The RNN cannot work on values in such a wide range. The embedding-layer is trained as
# a part of the RNN and will learn to map words with similar semantic meanings to similar embedding-vectors, 
# as will be shown further below.
model = Sequential()

# Adding the first layer 
model.add(Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_SIZE, input_length=max_tokens, name='layer_embedding'))

# First recurrent layer :  This will have 16 outputs. 
# Because we will add a second GRU after this one, 
# we need to return sequences of data because the next 
# GRU expects sequences as its input.
model.add(GRU(units=16, return_sequences=True))

# Second recurrent layer 
model.add(GRU(units=8, return_sequences=True))

# Third Recurrent Layer : This will be followed by a dense-layer,
# so it should only give the final output of the GRU and not a whole sequence of outputs
model.add(GRU(units=4))

# Add a fully-connected / dense layer which computes a value between 0.0 and 1.0 that will be used as the classification output.
model.add(Dense(1, activation='sigmoid'))

# Use the Adam optimizer with the given learning-rate.
optimizer = Adam(lr=1e-3)

# Compile 
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

logging.debug("Model Summary: \n%s" % (model.summary())) 


logging.info("Train the Recurrent Neural Network")


# ------------------- TRAINING ------------------- 
# Note that we are using the data-set with the padded sequences. 
# We use 5% of the training-set as a small validation-set,
#  so we have a rough idea whether the model is generalizing well 
# or if it is perhaps over-fitting to the training-set.
model.fit(x_train_pad, y_train,validation_split=0.05, epochs=3, batch_size=64) 



# ------------------- TEST SET PERFORMANCE -------------------
result = model.evaluate(x_test_pad, y_test)
logging.info("Accuracy: {0:.2%}".format(result[1]))



# ------------------- MORE EXAMPLES -------------------

text1 = "This movie is fantastic! I really like it because it is so good!"
text2 = "Good movie!"
text3 = "Maybe I like this movie."
text4 = "Meh ..."
text5 = "If I were a drunk teenager then this movie might be good."
text6 = "Bad movie!"
text7 = "Not a good movie!"
text8 = "This movie really sucks! Can I get my money back please?"
texts = [text1, text2, text3, text4, text5, text6, text7, text8]


tokens = tokenizer.texts_to_sequences(texts)
tokens_pad = pad_sequences(tokens, maxlen=max_tokens,padding=PAD, truncating=PAD)
logging.debug("Predictions: \n%s" % (model.predict(tokens_pad))) 


layer_embedding = model.get_layer('layer_embedding')
weights_embedding = layer_embedding.get_weights()[0]
token_good = tokenizer.word_index['good']
token_great = tokenizer.word_index['great']
token_bad = tokenizer.word_index['bad']
token_horrible = tokenizer.word_index['horrible']
logging.debug("Token Vector 'Good' : %s" % weights_embedding[token_good])
logging.debug("Token Vector 'Great' : %s" % weights_embedding[token_great])
logging.debug("Token Vector 'Bad' : %s" % weights_embedding[token_bad])
logging.debug("Token Vector 'Horrible' : %s" % weights_embedding[token_horrible])



def sorted_words(word, metric='cosine'):
    """
    Print the words in the vocabulary sorted according to their
    embedding-distance to the given word.
    Different metrics can be used, e.g. 'cosine' or 'euclidean'.
    """

    # Get the token (i.e. integer ID) for the given word.
    token = tokenizer.word_index[word]

    # Get the embedding for the given word. Note that the
    # embedding-weight-matrix is indexed by the word-tokens
    # which are integer IDs.
    embedding = weights_embedding[token]

    # Calculate the distance between the embeddings for
    # this word and all other words in the vocabulary.
    distances = cdist(weights_embedding, [embedding],
                      metric=metric).T[0]
    
    # Get an index sorted according to the embedding-distances.
    # These are the tokens (integer IDs) for words in the vocabulary.
    sorted_index = np.argsort(distances)
    
    # Sort the embedding-distances.
    sorted_distances = distances[sorted_index]
    
    # Sort all the words in the vocabulary according to their
    # embedding-distance. This is a bit excessive because we
    # will only print the top and bottom words.
    sorted_words = [inverse_map[token] for token in sorted_index
                    if token != 0]

    # Helper-function for printing words and embedding-distances.
    def _print_words(words, distances):
        for word, distance in zip(words, distances):
            print("{0:.3f} - {1}".format(distance, word))

    # Number of words to print from the top and bottom of the list.
    k = 10

    print("Distance from '{0}':".format(word))

    # Print the words with smallest embedding-distance.
    _print_words(sorted_words[0:k], sorted_distances[0:k])

    print("...")

    # Print the words with highest embedding-distance.
    _print_words(sorted_words[-k:], sorted_distances[-k:])

sorted_words('great', metric='cosine')
sorted_words('worst', metric='cosine')