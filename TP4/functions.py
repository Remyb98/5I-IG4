import glob
import cv2
import numpy as np


def warn(*args, **kwargs): pass


import warnings

warnings.warn = warn
from sklearn.cluster import MiniBatchKMeans
import pickle
import os
import dml
from sklearn.svm import SVC

# https://yohanes.gultom.id/2018/05/20/sift-surf-bow-for-big-number-of-clusters/
import random


def vocabulaire(N: int, paths: list):
    descriptors_list = list()
    s = cv2.xfeatures2d.SURF_create()
    for path in paths:
        for image in glob.glob(f'{path}/*.jpg'):
            current_image = cv2.imread(image)
            current_image_but_in_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            kpts, descriptors = s.detectAndCompute(current_image_but_in_gray, None)
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
        if local_max > error: error = local_max
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


def test_vect(classes, model):
    vect_list = []
    file_list = []
    for path in classes:
        for image in glob.glob(f'{path}/*.jpg'):
            vect_list.append(vectoriser(cv2.imread(image), model))
            file_list.append(image)

    with open(os.path.join('saves', 'base_vectors.pickle'), 'wb') as f:
        pickle.dump(vect_list, f)
    with open(os.path.join('saves', 'base_files.pickle'), 'wb') as f:
        pickle.dump(file_list, f)


def apprentissage(N, vectors=None, files=None):
    classes = ['flamingo', 'panda']
    if not vectors and not files:
        vectors = pickle.load(open(os.path.join('saves', 'base_vectors.pickle'), 'rb'))
        files = pickle.load(open(os.path.join('saves', 'base_files.pickle'), 'rb'))

    X = []
    Y = []
    for i in range(len(vectors)):
        vect = vectors[i]
        file = files[i]
        for j, c in enumerate(classes):
            if c in file:
                Y.append(j + 1)
                temp = []
                for _ in range(N):
                    temp.append(0)
                for e in vect:
                    temp[e] += 1
                X.append(temp)
    return X, Y


def KDA(X, Y):
    s = dml.kda.KDA(n_components=2, kernel='poly', degree=8)
    s.fit(X, Y)
    temp_value = s.transform(X)
    print(str("\nDegré polynome : " + str(8) + " -- Valeurs transform : " + str(temp_value)))


def load_test_data(model):
    classes = ['./classes/flamingo/test', './classes/panda/test']
    vect_list = list()
    file_list = list()
    for path in classes:
        for image in glob.glob(f'{path}/*.jpg'):
            vect_list.append(vectoriser(cv2.imread(image), model))
            file_list.append(image)
    return apprentissage(len(model.classes_))


def learn_svc(X, Y, model: MiniBatchKMeans):
    model = SVC(C=2, kernel="poly", degree=2)
    model.fit(X, Y)
    X_test, Y_test = load_test_data(model)
    # res = model.predict(X_test)

    # print(res)
    print(Y_test)
