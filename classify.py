import utility
import os
import sys
import cv2
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train model with previously generated codebook")
    parser.add_argument('-c', '--cluster', help="Clustering strategy", required=True)
    parser.add_argument('-f', '--feature', help="Extract feature strategy", required=True)
    parser.add_argument('-m', '--model', help="Model generated", required=True)
    args = vars(parser.parse_args())

    images = utility.open_image_folder(os.getcwd() + "/test")
    if (args['cluster'] is not None and args['feature'] is not None and args['model'] is not None):
        with open("codebook_" + args['feature'].lower() + '_' + args['cluster'].lower() + ".pickle", 'rb')  as handle_codebook, \
        open("model_" + args['model'].lower() + ".pickle", "rb") as handle_model:
            cluster_alg = pickle.load(handle_codebook)
            classifier = pickle.load(handle_model)
    else:
        sys.exit()

    extractor = None
    if (args['feature'] == 'sift'):
        extractor = cv2.xfeatures2d.SIFT_create()

    for key in images:
        print(key)
        for image in images[key]:
            image = utility.gray(image)
            keypoint, descriptor = utility.features(image, extractor)
            histogram = utility.build_histogram(descriptor,cluster_alg)
            print(classifier.predict([histogram]))
