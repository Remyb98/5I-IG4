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
    model.fit(dataset)
    np.savetxt('cluster_center.txt', model.cluster_centers_, delimiter=',')
    variance = model.inertia_
    norm = random.randint(0, 15)
    return variance, norm
