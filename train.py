import pickle
import utility
import os
import cv2
from sklearn.neural_network import MLPClassifier
import time

if __name__ == "__main__":
    cluster_model = None
    with open("codebook.pickle", 'rb') as handle:
        cluster_model = pickle.load(handle)

    data_images = utility.open_image_folder(os.getcwd() + "/images")
    preprocessed_image = []
    target = []
    extractor = cv2.xfeatures2d.SIFT_create()
    for key in data_images:
        for image in data_images[key]:
            image = utility.gray(image)
            keypoint, descriptor = utility.features(image, extractor)
            histogram = utility.build_histogram(descriptor, cluster_model)
            preprocessed_image.append(histogram)
            target.append(key)

    classifier = MLPClassifier(batch_size=30)
    print("Start learning...")
    start = time.time()
    classifier.fit(preprocessed_image, target)
    print("Time to learn: %s " % (time.time() - start))

    with open("model.pickle", "wb") as handle:
        pickle.dump(classifier,handle)
