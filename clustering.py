import argparse
import numpy as np
import cv2

class ColorClustering:
    def __init__(self):
        self.K = 16

    def run(self, src, dst):
        src_img = cv2.imread(src)
        samples = np.float32(src_img.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts = 3
        compactness, labels, centers = cv2.kmeans(samples, self.K, criteria, attempts, cv2.KMEANS_PP_CENTERS)

        centers = np.uint8(centers)
        dst_img = centers[labels.flatten()].reshape(src_img.shape)
        cv2.imwrite(dst, dst_img)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', required=True)
    parser.add_argument('-d', '--dst', required=True)
    args = parser.parse_args()

    color_clustering = ColorClustering()

    color_clustering.run(args.src, args.dst)


