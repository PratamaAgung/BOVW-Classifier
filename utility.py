import cv2
import os
import glob
import numpy as np

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

def open_image_folder(image_folder):
    image_list = {}
    count_image = 0
    for folder in glob.glob(image_folder + "/*"):
        image_class = folder.split("/")[-1]
        image_list[image_class] = []
        for image in glob.glob(folder + "/*.[Jj][Pp][Gg]"):
            read_image = cv2.imread(image)
            image_list[image_class].append(read_image)
            count_image += 1
    print("Read %s image" % str(count_image))
    return image_list

def formatND(l):
    vStack = np.vstack(l)
    return vStack
