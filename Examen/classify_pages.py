# -*- coding: utf-8 -*-
"""
CLASSIFY PAGES

G. Audiberti - 2018
"""

import argparse
import logging
import sys
import time
import os
import pandas as pd
from sklearn import neighbors, linear_model, metrics
from tqdm import tqdm
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

IMG_FEATURE_SIZE = (12, 16)

# Setup logging
logger = logging.getLogger('renameIt.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def extract_features(img):
    """
    Compute the subresolution of an image and return it as a feature vector
    :param img: the original image (can be color or gray)
    :type img: pillow image
    :return: pixel values of the image in subresolution
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
        IMG_FEATURE_SIZE, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]

def train_test_classifier(clf,X_train,y_train,X_test,y_test):
    # Do classification
    t0 = time.time()
    logger.info("Training...")
    clf.fit(X_train,y_train)
    logger.info("Done in %0.3fs" % (time.time() - t0))
    predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, predicted))
    test_acc = metrics.accuracy_score(y_test, predicted)
    return (test_acc)

def hyperoptize_knn(X_train,y_train,X_test,y_test):
    # Create validation set so that train = 60% , validation = 20% and test =  20%
    X_train_hyper, X_valid_hyper, y_train_hyper, y_valid_hyper = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    for k in [1,2,3,4,5,6,7,8,9,10]:
        logger.info("k={}".format(k))
        best_k = 0
        best_acc = 0
        clf = neighbors.KNeighborsClassifier(k)
        clf.fit(X_train_hyper,y_train_hyper)

        for _name,_train_set,_test_set in [('train',X_train_hyper,y_train_hyper),('valid',X_valid_hyper,y_valid_hyper),('test',X_test,y_test)]:
            _predicted = clf.predict(_train_set)
            _accuracy = metrics.accuracy_score(_test_set, _predicted)
            if (_accuracy > best_acc):
                best_k = k
                best_acc = _accuracy
            logger.info("{} accuracy : {}".format(_name,_accuracy))

    print('Best k parameter for KNN is : '+str(best_k)+' with '+str(best_acc)+' of accuracy score')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on images and test the classifier')
    required_inputs = parser.add_mutually_exclusive_group(required=True)
    required_inputs.add_argument('--images-list',help='file containing the image path and image class, one per line, comma separated')
    required_inputs.add_argument('--load-features',help='read features and class from pickle file')
    parser.add_argument('--save-features',help='save features in pickle format')
    parser.add_argument('--limit-samples',type=int, help='limit the number of samples to consider for training')
    parser.add_argument('--subresolution',type=int, default=8,help='value for ....')
    parser.add_argument('--scale-features',action='store_true',default=False)

    classifier_group = parser.add_mutually_exclusive_group()
    classifier_group.add_argument('--nearest-neighbors', type=int)
    classifier_group.add_argument('--optimize-nearest-neighbors', action='store_true')
    classifier_group.add_argument('--logistic-regression', action='store_true')

    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--features-only', action='store_true', help='only extract features, not classification')
    action_group.add_argument('--classify', action='store_true')
    action_group.add_argument('--learning-curve', action='store_true')
    action_group.add_argument('--testing-curve', action='store_true')

    args = parser.parse_args()

    print('Starting...');

    if args.save_features:

        all_df = pd.read_csv(args.images_list, header=None, names=['file_path', 'type', 'name'], sep=',')
        logger.info('Loaded {} path in {}'.format(all_df.shape, args.images_list))
        # print(all_df);

        data = []

        file_list = all_df['file_path']
        y = all_df['type']

        for i_path in tqdm(file_list):
            if os.path.exists(i_path):
                page_image = Image.open(i_path)
                data.append(extract_features(page_image))
                # print(i_path)

        X = pd.np.array(data)

        df_features = pd.DataFrame(X)
        df_features['type'] = y
        df_features.to_pickle(args.save_features)
        logger.info('Saved {} features and type to {}'.format(df_features.shape,args.save_features))
        # print(X)

    if args.load_features:

        df_features = pd.read_pickle(args.load_features)
        logger.info('Loaded {} features'.format(df_features.shape))
        if args.limit_samples:
            df_features = df_features.sample(n=args.limit_samples)
        if 'type' in df_features.columns:
            X = df_features.drop(['type'], axis=1)
            y = df_features['type']
        else:
            logger.error('Can not find classes in pickle')
            sys.exit(1)
        # print(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Train a classifier
    if args.classify:

        if args.nearest_neighbors:
            clf = neighbors.KNeighborsClassifier(args.nearest_neighbors)
            clf_name = "{}NN".format(args.nearest_neighbors)
        elif args.optimize_nearest_neighbors:
            clf_name = "NN"
            hyperoptize_knn(X_train, y_train, X_test, y_test)
        elif args.logistic_regression:
            clf = linear_model.LogisticRegression()
            clf_name = "LogReg"
        else:
            logger.error('No classifier specified')
            sys.exit()
        logger.info("Classifier is {}".format(clf_name))

        train_acc = train_test_classifier(clf,X_train, y_train, X_test, y_test)
        print(train_acc)
