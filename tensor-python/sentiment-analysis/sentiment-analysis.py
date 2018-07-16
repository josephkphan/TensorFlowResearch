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


# ------------------- EMBEDDING ------------------- #
logging.info("STARTING PHASE - EMBEDDING") 