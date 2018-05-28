import pickle
import utility
import os
import sys
import cv2
from sklearn.neural_network import MLPClassifier
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train model with previously generated codebook")
    parser.add_argument('-c', '--cluster', help="Clustering strategy", required=True)
    parser.add_argument('-f', '--feature', help="Extract feature strategy", required=True)
    parser.add_argument('-m', '--model', help="Model generated", required=True)
    args = vars(parser.parse_args())

    cluster_model = None
    if (args['feature'] is not None and args['cluster'] is not None):
        with open("codebook_" + args['feature'].lower() + '_' + args['cluster'].lower() + ".pickle", 'rb') as handle:
            cluster_model = pickle.load(handle)
    else:
        sys.exit()

    extractor = None
    if (args['feature'].lower() == 'sift'):
        extractor = cv2.xfeatures2d.SIFT_create()
    elif (args['feature'].lower() == 'kaze'):
        extractor = cv2.KAZE_create()
    elif (args['feature'].lower() == 'orb'):
        extractor = cv2.ORB_create()
    else:
        sys.exit()

    classifier = None
    if (args['model'].lower() == 'mlp'):
        classifier = MLPClassifier(batch_size=30)
    else:
        sys.exit()

    data_images = utility.open_image_folder(os.getcwd() + "/images")
    preprocessed_image = []
    target = []
    for key in data_images:
        for image in data_images[key]:
            image = utility.gray(image)
            keypoint, descriptor = utility.features(image, extractor)
            histogram = utility.build_histogram(descriptor, cluster_model)
            preprocessed_image.append(histogram)
            target.append(key)

    print("Start learning...")
    start = time.time()
    classifier.fit(preprocessed_image, target)
    print("Time to learn: %s " % (time.time() - start))

    with open("model_" + args['model'].lower() + '_' + args['feature'].lower() + '_' + args['cluster'].lower() + ".pickle", "wb") as handle:
        pickle.dump(classifier,handle)
