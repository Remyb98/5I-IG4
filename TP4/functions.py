import glob
import cv2
import numpy as np
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
from sklearn.cluster import MiniBatchKMeans


# https://yohanes.gultom.id/2018/05/20/sift-surf-bow-for-big-number-of-clusters/
import random
def vocabulaire(N : int, paths : list):
    descriptors_list = list()
    s = cv2.xfeatures2d.SURF_create()
    for path in paths:
        for image in glob.glob(f'{path}/*.jpg'):
            current_image = cv2.imread(image)
            current_image_but_in_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            kpts, descriptors = s.detectAndCompute(current_image_but_in_gray, None)
            # kp_img = cv2.drawKeypoints(current_image_but_in_gray, kpts, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            descriptors_list.extend(descriptors)
    dataset = np.array(descriptors_list)
    model = MiniBatchKMeans(n_clusters=N)
    predicted = model.fit_predict(dataset)
    variance = model.inertia_
    centers = model.cluster_centers_
    np.savetxt('cluster_center.txt', centers, delimiter=',')
    error = 0
    for i in range(len(predicted)):
        class_predicted = predicted[i]
        center = centers[class_predicted]
        local_center = dataset[i]
        local_max = np.linalg.norm(center - local_center)
        if(local_max > error): error = local_max
    return variance, error, model


def vectoriser(image, vocabulaire):
    s = cv2.xfeatures2d.SURF_create()
    image_but_in_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kpts, descriptors = s.detectAndCompute(image_but_in_gray, None)
    descriptors_list = np.array([])
    descriptors_list = np.append(descriptors_list, descriptors)

    descriptors = np.reshape(descriptors_list, (len(descriptors_list) // 64, 64))
    descriptors = np.float32(descriptors)

    return vocabulaire.predict(descriptors)

def testVect(classes, model):
    vect_list = []
    for path in classes:
        for image in glob.glob(f'{path}/*.jpg'):
            vect_list.append(vectoriser(cv2.imread(image), model))
    return vect_list
