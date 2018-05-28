import utility
from sklearn.cluster import MiniBatchKMeans, DBSCAN
import cv2
import os
import time
import pickle

if __name__ == '__main__':
    data_image = utility.open_image_folder(os.getcwd() + '/images')
    descriptor_list = []
    extractor = cv2.xfeatures2d.SIFT_create()

    start = time.time()
    for key in data_image:
        for image in data_image[key]:
            gray_image = utility.gray(image)
            keypoints, descriptors = utility.features(gray_image, extractor)
            descriptor_list.append(descriptors)
    print("Extracting descriptor time: %s" % (time.time() - start))

    start = time.time()
    vocab_size = 20
    cluster_alg = MiniBatchKMeans(n_clusters = vocab_size, batch_size=25)
    descriptor_numpy = utility.formatND(descriptor_list)
    cluster_alg.fit_predict(descriptor_numpy)
    print("Clustering time: %s" % (time.time() - start))

    with open("codebook.pickle", 'wb') as handle:
        pickle.dump(cluster_alg, handle)
