import utility
from sklearn.cluster import MiniBatchKMeans, DBSCAN
import cv2
import os
import time
import pickle
import argparse
import sys

if __name__ == '__main__':
    data_image = utility.open_image_folder(os.getcwd() + '/images')
    descriptor_list = []

    parser = argparse.ArgumentParser("Build codebook with some clustering strategy and output it into a file")
    parser.add_argument('-c', '--cluster', help="Clustering strategy", required=True)
    parser.add_argument('-n', '--nb_cluster', help="Number of cluster used", required=False)
    parser.add_argument('-f', '--feature', help="Extract feature strategy", required=True)
    args = vars(parser.parse_args())

    dictionary_size = 20
    if (args['nb_cluster'] is not None):
        dictionary_size = int(args['nb_cluster'])

    cluster_alg = None
    if (args['cluster'].lower() == 'kmeans'):
        cluster_alg = MiniBatchKMeans(n_clusters = dictionary_size, batch_size=25)
    else:
        sys.exit()

    extractor = None
    if (args['feature'].lower() == 'sift'):
        extractor = cv2.xfeatures2d.SIFT_create()
    else:
        sys.exit()

    start = time.time()
    for key in data_image:
        for image in data_image[key]:
            gray_image = utility.gray(image)
            keypoints, descriptors = utility.features(gray_image, extractor)
            descriptor_list.append(descriptors)
    print("Extracting descriptor time: %s" % (time.time() - start))

    start = time.time()
    descriptor_numpy = utility.formatND(descriptor_list)
    cluster_alg.fit_predict(descriptor_numpy)
    print("Clustering time: %s" % (time.time() - start))

    with open("codebook_" + args['feature'].lower() + '_' + args['cluster'].lower() + ".pickle", 'wb') as handle:
        pickle.dump(cluster_alg, handle)
