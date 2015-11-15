import argparse
import numpy as np
import cv2

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', required=True)
    parser.add_argument('-d', '--dst', required=True)
    args = parser.parse_args()

    src = cv2.imread(args.src)
    samples = np.float32(src.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 16
    attempts = 3
    compactness, labels, centers = cv2.kmeans(samples, K, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)
    dst = centers[labels.flatten()].reshape(src.shape)
    cv2.imwrite(args.dst, dst)
