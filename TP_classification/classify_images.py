# -*- coding: utf-8 -*-
"""
Classify digit images

C. Kermorvant - 2017
"""


import argparse
import logging
import time
import sys

from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn import svm, metrics, neighbors
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plot

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

# Train and print metrics for a given classifier with training set and test test
# Return train accuracy score and test accuracy score
def classifier_train_test(classifier_name, classifier,X_train, X_test, y_train, y_test) :

    # Do Training@
    t0 = time.time()
    clf.fit(X_train, y_train)
    logger.info("Training done in %0.3fs" % (time.time() - t0))

    # Do testing
    logger.info("Testing "+classifier_name)
    t0 = time.time()
    predicted_test = classifier.predict(X_test)

    logger.info("Testing  done in %0.3fs" % (time.time() - t0))
    #print('Score on testing : %f' % classifier.score(X_test, y_test))
    return classifier.score(X_train, y_train),metrics.accuracy_score(y_test,predicted_test)

def getAccuracyTesting(X_test, Y_test, sizeTest):
    X_none, X_test, Y_none, Y_test = train_test_split(X_test, Y_test, test_size=sizeTest)
    clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    return metrics.accuracy_score(Y_test, predicted)

def getAccuracy(X_train, Y_train, sizeTrain):
    X_train, X_none, Y_train, y_none = train_test_split(X_train, Y_train, train_size=sizeTrain)
    clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    return metrics.accuracy_score(Y_test, predicted), clf.score(X_train, Y_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-list',help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features',help='read features and class from pickle file')
    parser.add_argument('--save-features',help='save features in pickle format')
    parser.add_argument('--limit-samples',type=int, help='limit the number of samples to consider for training')
    parser.add_argument('--learning-curve', action='store_true')
    parser.add_argument('--testing-curve', action='store_true')
    classifier_group = parser.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('--nearest-neighbors',type=int)
    classifier_group.add_argument('--logistic-regression',type=int)
    classifier_group.add_argument('--nearest-neighbors-logistic-regression', action='store_true')
    classifier_group.add_argument('--kernelRBF', action='store_true')
    classifier_group.add_argument('--features-only', action='store_true', help='only extract features, do not train classifiers')

    args = parser.parse_args()

    if args.load_features:
        # read features from indicated pickle file
        df = pd.read_pickle(args.load_features)
        #limit the dataframe to X samples
        if args.limit_samples:
            df = df.sample(args.limit_samples)
        y = list(df["class"])
        X = df.drop(columns='class')
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

    if args.nearest_neighbors:
        # create KNN classifier with args.nearest_neighbors as a parameter
        clf = KNeighborsClassifier(args.nearest_neighbors)
        logger.info('Use kNN classifier with k= {}'.format(args.nearest_neighbors))

        # Use train_test_split to create train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))

        train_accuracy, test_accuracy = classifier_train_test("KNN", clf, X_train, X_test, Y_train, Y_test)
        print('KNN Train accuracy score : ', train_accuracy)
        print('KNN Test accuracy score :', test_accuracy)

    elif args.logistic_regression:
        clf = LogisticRegression()
        logger.info('Use logistic regression classifier with k= {}'.format(args.logistic_regression))

        # Use train_test_split to create train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))

        train_accuracy, test_accuracy = classifier_train_test("Logistic Regression", clf, X_train, X_test, Y_train, Y_test)
        print('Logistic Train accuracy score :', train_accuracy)
        print('Logistic Test accuracy score :', test_accuracy)

    elif args.nearest_neighbors_logistic_regression:
        # Use train_test_split to create train/test split
        X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, train_size=0.8)
        logger.info("Test set size is {}".format(X_test.shape))
        # Use train_test_split to create train/validation split
        X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, train_size=0.8)
        logger.info("Training set size is {}".format(X_train.shape))
        logger.info("Validation set size is {}".format(X_validation.shape))

        # Select best KNN classifier k
        best_k = [0, 0]
        for i in range(1, 10):
            clf = neighbors.KNeighborsClassifier(i)
            logger.info('Use kNN classifier with k= {}'.format(i))
            # Do Training and Testing
            train_accuracy, test_accuracy = classifier_train_test("KNN", clf, X_train, X_validation, y_train, y_validation)
            print('Knn accuracy score :', test_accuracy)
            if (test_accuracy > best_k[1]):
                best_k[0] = i
                best_k[1] = test_accuracy

        # Execute classifier
        print('Best k = {}'.format(best_k[0]))
        print('Validation accuracy score = {}'.format(best_k[1]))

    elif args.kernelRBF:
        # Use train_test_split to create train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))

        # LINEAR
        # clf = SVC(kernel='linear')

        # RBF
        param = {'C': [0.5, 1, 5], 'gamma': [0.05, 0.1, 0.5]}
        svc = svm.SVC(kernel='rbf')
        clf = GridSearchCV(svc, param)

        train_accuracy, test_accuracy = classifier_train_test("RBF", clf, X_train, X_test, Y_train, Y_test)
        print('RBF Train accuracy score : ', train_accuracy)
        print('RBF Test accuracy score :', test_accuracy)

    else:
        logger.error('No classifier specified')
        sys.exit()


if args.learning_curve:
     curv_y = []
     curv_y2 = []
     training_size = np.array([0.01, 0.10, 0.20, 0.40, 0.60, 0.80, 0.99])
     curv_x = df.shape[0] * np.array(training_size)
     for i in range(0, 7):
         curv_y.append(getAccuracy(X_train, Y_train, training_size[i])[0])
         curv_y2.append(getAccuracy(X_train, Y_train, training_size[i])[1])
     plot.plot(curv_x, curv_y)
     plot.plot(curv_x, curv_y2)
     plot.title("Training curves")
     plot.xlabel("Train set size")
     plot.ylabel("Accuracy")
     plot.show()

if args.testing_curve:
 accuracy_score = []
 means = []
 errors = []
 testing_size = np.array([0.01, 0.10, 0.20, 0.40, 0.60, 0.80, 0.99])
 test_set_size = df.shape[0] * np.array(testing_size)
 for i in range(0, 7):
     for j in range(0, 10):
         accuracy_score.append(getAccuracyTesting(X_test, Y_test, testing_size[i]))
     means.append(np.mean(accuracy_score))
     errors.append(np.std(accuracy_score))
     print("Mean of test set = {} for a test set of size {}".format(np.mean(accuracy_score), testing_size[i]))
     print("Standard deviation of test set = {} for a test set of size {}".format(np.std(accuracy_score), testing_size[i]))
 plot.scatter(test_set_size, means)
 plot.errorbar(test_set_size, means, errors, ecolor='red')
 plot.title("Testing curves")
 plot.xlabel("Test set size")
 plot.ylabel("Test accuracy")
 plot.show()
