# -*- coding: utf-8 -*-
"""
Classify digit images

C. Kermorvant - 2017
"""


import argparse
import logging
import time
import sys

from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn import svm, metrics, neighbors
from sklearn.model_selection import train_test_split

import numpy as np


# Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def extract_features_subresolution(img,img_feature_size = (8, 8)):
    """

    For a given image, function computes a 8x8 subresolution image and
    returns a feature vector with pixel values.
    (0 must correspond to white and 255 to black)

    :param img: the original image (can be color or gray)
    :type img: pillow image
    :return: pixel values - feature vector
    :rtype: list of int in [0,255]

    """

    # convert color images to grey level
    gray_img = img.convert('L')
    # find the min dimension to rotate the image if needed
    min_size = min(img.size)
    if img.size[1] == min_size:
        # convert landscape  to portrait
        rotated_img = gray_img.rotate(90, expand=1)
    else:
        rotated_img = gray_img

    # reduce the image to a given size
    reduced_img = rotated_img.resize(
        img_feature_size, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-list',help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features',help='read features and class from pickle file')
    parser.add_argument('--save-features',help='save features in pickle format')
    parser.add_argument('--limit-samples',type=int, help='limit the number of samples to consider for training')
    classifier_group = parser.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('--nearest-neighbors',type=int)
    classifier_group.add_argument('--features-only', action='store_true', help='only extract features, do not train classifiers')
    args = parser.parse_args()


    if args.load_features:
        # read features from indicated pickle file
        df = pd.read_pickle(args.load_features)
        y = list(df["class"])
        #print(df)
        #print(y)
        pass
    else:

        # Load the image list from CSV file using pd.read_csv
        # see the doc for the option since there is no header ;
        # specify the column names :  filename , class
        file_list = []
        column_names = ['filename', 'class']
        file_list = pd.read_csv(args.images_list, header=None, names=column_names)
        logger.info('Loaded {} images in {}'.format(file_list.shape,args.images_list))
        #print(file_list)

        # Extract the feature vector on all the pages found
        # Modify the extract_features from TP_Clustering to extract 8x8 subresolution values
        # white must be 0 and black 255
        data = []
        for i_path in tqdm(file_list.filename):
            page_image = Image.open(i_path)
            data.append(extract_features_subresolution(page_image))

        # check that we have data
        if not data:
            logger.error("Could not extract any feature vector or class")
            sys.exit(1)

        # convert to np.array
        X = np.array(data)
        y = file_list["class"]


    # save features
    if args.save_features:
        # convert X to dataframe with pd.DataFrame
        df = pd.DataFrame(X)
        df["class"] = y
        # and save to pickle with to_pickle
        df.to_pickle(args.save_features+'.pickle')
        logger.info('Saved {} features and class to {}'.format(df.shape, args.save_features))


    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()

    # Train classifier
    logger.info("Training Classifier")

    # Use train_test_split to create train/test split
    logger.info("Train set size is {}".format(X_train.shape))
    logger.info("Test set size is {}".format(X_test.shape))

    if args.nearest_neighbors:
        # create KNN classifier with args.nearest_neighbors as a parameter
        logger.info('Use kNN classifier with k= {}'.format(args.nearest_neighbors))
    else:
        logger.error('No classifier specified')
        sys.exit()

    # Do Training@
    t0 = time.time()
    logger.info("Training  done in %0.3fs" % (time.time() - t0))

    # Do testing
    logger.info("Testing Classifier")
    t0 = time.time()
    predicted = clf.predict(X_test)

    # Print score produced by metrics.classification_report and metrics.accuracy_score
    logger.info("Testing  done in %0.3fs" % (time.time() - t0))
