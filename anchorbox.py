import numpy as np
import random
class anchorbox:
    def __init__(self, k=5, box=[]):
        self.k = k
        self.box = box
        self.box = np.array(box, dtype = np.float32)
        self.centroids = np.array(random.sample(self.box, k))
        self.dist = np.zeros([self.box,k])
        self.idx = np.empty(len(self.box))

    def IOU(self, box):
        iou = []
        for centroid in self.centroids:
            w_min = np.minimum(box[0], self.centroid[0])
            h_min = np.minimum(box[1], self.centroid[1])
            
            overlap = w_min*h_min
            iou = 1 - (overlap / (box[0]*box[1] + self.centroid[0]*self.entroid[1] - overlap))

        iou.append(iou)
        return np.array(iou)
    
    def set_cluster(self):
        for i, box in enumerate(self.box):
            dist = self.IOU(box)
            self.idx[i] = np.argmin(dist,axis=1)

    def update_centroids(self):
        for i in range(self.k):
            idx = self.idx == i
            kth_boxes = self.centroids[idx]
            next_centroids = np.mean(kth_boxes,axis=1)
            self.centroids[i,:] = next_centroids

    def kmeans(self):
        while True:

