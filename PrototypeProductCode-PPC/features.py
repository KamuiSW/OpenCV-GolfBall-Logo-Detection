import traceback
import cv2
import numpy as np


def createDetector():
    detector = cv2.ORB_create(nfeatures=5000)
    return detector

def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = createDetector()
    kps, descs = detector.detectAndCompute(gray, None)
    return kps, descs, img.shape[:2][::-1]


def detectFeatures(img, train_features):
    train_kps, train_descs, shape = train_features
    # get features from input image
    kps, descs, _ = getFeatures(img)
    # check if keypoints are extacted
    if not kps:
        return None
  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(train_descs, descs, k=2)

    good = []
    try:
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append([m])

        # stop if not enough matching shit :(
        if len(good) < 0.08 * len(train_kps):
            return None

        # estimate matrix and some incredibly weird stuff happening idk
        src_pts = np.float32([train_kps[m[0].queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m[0].trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if m is not None:
            # apply perspective to trainnn
            scene_points = cv2.perspectiveTransform(np.float32([(0, 0), (0, shape[0] - 1),
                                                                (shape[1] - 1, shape[0] - 1),
                                                                (shape[1] - 1, 0)]).reshape(-1, 1, 2), m)
            rect = cv2.minAreaRect(scene_points)
            # check le result :D
            if rect[1][1] > 0 and 0.7< (rect[1][0] / rect[1][1]) < 1.2:
                return rect
    except:
        pass
    return None