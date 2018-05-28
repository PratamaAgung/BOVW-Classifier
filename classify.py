import utility
import os
import cv2
import pickle

if __name__ == '__main__':
    images = utility.open_image_folder(os.getcwd() + "/test")
    extractor = cv2.xfeatures2d.SIFT_create()
    with open("codebook.pickle", 'rb')  as handle_codebook, open("model.pickle", 'rb') as handle_model:
        cluster_alg = pickle.load(handle_codebook)
        classifier = pickle.load(handle_model)

    for key in images:
        print(key)
        for image in images[key]:
            image = utility.gray(image)
            keypoint, descriptor = utility.features(image, extractor)
            histogram = utility.build_histogram(descriptor,cluster_alg)
            print(classifier.predict([histogram]))
