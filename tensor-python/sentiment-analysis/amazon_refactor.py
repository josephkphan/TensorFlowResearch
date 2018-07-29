import argparse
import json
import os, sys
import pprint
import math

pp = pprint.PrettyPrinter(indent=4)

"""
Example Execution: (executing from /${ROOT_OF_REPO}/tensor-python/sentiment-analysis/)
    python3 amazon_refactor.py -d data/AMAZON/musical-instruments/ -f reviews_Musical_Instruments_5.json

Prerequistes: data/AMAZON/musicial-instruments/reviews_Musical_Instruments_5.json needs to exist. 
    Download Amazon jsons from here: http://jmcauley.ucsd.edu/data/amazon/ 
"""
# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument( '-d', '--dir', type=str, help='path to directory  with json file to parse ENDING in /', required=True)
parser.add_argument( '-f', '--file', type=str, help='filename of json file inside specified directory', required=True)
args = parser.parse_args()


"""
Function Description: 
    * Creates the expected directory/file structure expected by the ML Program
        Example:

        Expects:
            `-- data
                `-- AMAZON
                    `-- ./musical-instruments
                        |-- ./musical-instruments/reviews_Musical_Instruments_5.json

        Creates:
            `-- data
                `-- AMAZON
                    `-- ./musical-instruments
                        |-- ./musical-instruments/reviews_Musical_Instruments_5.json
                        |-- ./musical-instruments/test
                        |   |-- ./musical-instruments/test/neg
                        |   `-- ./musical-instruments/test/pos
                        `-- ./musical-instruments/train
                            |-- ./musical-instruments/train/neg
                            `-- ./musical-instruments/train/pos%

"""
def create_directory_structure():
    try:
        os.mkdir(args.dir + 'train')
        os.mkdir(args.dir + 'train/pos')
        os.mkdir(args.dir + 'train/neg')
        os.mkdir(args.dir + 'test')
        os.mkdir(args.dir + 'test/pos')
        os.mkdir(args.dir + 'test/neg')

    except Exception as e: 
        print ('Directory structure already exists')

"""
Function Description:
    * Reads through the json file
    * ID each review
    * Calculates sentiment value via floor(2x 5-star-Review) (for a sentiment value from 0-10)
    * Returns a dictionary with all reviews
        Example content : 
            dict = {
                "<REVIEW_ID>": {
                    "text": "<REVIEW_TEXT>"
                    "value": "<REVIEW_VALUE>"
                },
                ...
            }
"""
def parse_json():
    reviews = {}
    with open(args.dir + args.file) as infile:
        reviewID = 0
        for line in infile:
            reviews[reviewID] = {}
            review = json.loads(line)
            reviews[reviewID]['text'] = review['reviewText']
            reviews[reviewID]['value'] = int(review['overall'] * 2 )
            reviewID += 1
            # if reviewID > 50:
            #    break
    pp.pprint(reviews)
    return reviews, reviewID


"""
Function Description:
    * Breaks up reviews into testing and training (~5% as testing)
    * Creates individual files for each review with filename <REVIEW_ID>_<VALUE>.txt and 
        the contents of each file the text of that particular review
"""
def create_data_sets(reviews, num_reviews):
    def write_review_to_file(file_path, review_text):
        with open(file_path, 'w') as f:  
            f.write(review_text)

    def write_statistics_json_file(file_path, json_dict): 
        out_file = open(file_path, "w")
        json.dump(json_dict, out_file, indent=4)
        out_file.close()

    testing_cutoff = int( math.ceil(.05 * num_reviews) )
    test_pos_counter = 0
    test_neg_counter = 0
    test_sum = 0 
    train_pos_counter = 0
    train_neg_counter = 0
    train_sum = 0


    # Testing Data
    for i in range(0, testing_cutoff):
        # Determining whether review is a positive or negative review
        file_name = str(i) + '_' + str(reviews[i]['value']) + '.txt'
        if reviews[i]['value'] >= 7:
            sub_dir = 'pos/'
            test_pos_counter += 1
        else:
            sub_dir = 'neg/'
            test_neg_counter += 1
        test_sum += reviews[i]['value']
        
        # Write review to file
        file_path = args.dir + 'test/' + sub_dir + file_name
        write_review_to_file(file_path, reviews[i]['text'])

    # Training Data
    for i in range(testing_cutoff + 1, num_reviews):
        # Determining whether review is a positive or negative review
        file_name = str(i) + '_' + str(reviews[i]['value']) + '.txt'
        if reviews[i]['value'] >= 7:
            sub_dir = 'pos/'
            train_pos_counter += 1
        else:
            sub_dir = 'neg/'
            train_neg_counter += 1
        train_sum += reviews[i]['value']
        
        # Write review to file
        file_path = args.dir + 'train/' + sub_dir + file_name
        write_review_to_file(file_path, reviews[i]['text'])

    # Write statistics of reviews to file
    statistics = {
        "test_pos_count": test_pos_counter,
        "test_neg_count": test_neg_counter,
        "test_avg": (test_sum / testing_cutoff),
        "train_pos_count": train_pos_counter,
        "train_neg_count": train_neg_counter,
        "train_avg": (train_sum / num_reviews - testing_cutoff)
    }
    write_statistics_json_file(args.dir + 'stats.json', statistics)


if __name__== "__main__":
    create_directory_structure()
    reviews, num_reviews = parse_json()
    create_data_sets(reviews, num_reviews)