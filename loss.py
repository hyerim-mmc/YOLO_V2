import torch
import numpy as np
import torch.nn as nn
import utils


def yolov2_loss():
    anchor = np.load("./dataset/anchor.npy")
    anchor_w, anchor_h = anchor[:, 0], anchor[:, 1]

    ground_truth = []
    grid = []

    for i in range(13):
        for j in range(13):
            for b in range(5):
                coord_scale = 5
                obj_scale = 0
                noobj_scale = 0.5

                true_x[i, j, b] = ground_truth.center_x - grid[i, j].center_x
                true_y[i, j, b] = ground_truth.center_y - grid[i, j].center_y
                true_w[i, j, b] = np.logspace(ground_truth.width / anchor_w[b])
                true_h[i, j, b] = np.logspace(ground_truth.height / anchor_h[b])

                pred_x[i, j, b] = 0
                pred_y[i, j, b] = 0
                pred_w[i, j, b] = 0
                pred_h[i, j, b] = 0

                box_x[i, j, b] = (i + nn.Sigmoid(pred_x[i, j, b])) * 32
                box_y[i, j, b] = (j + nn.Sigmoid(pred_y[i, j, b])) * 32
                box_w[i, j, b] = anchor_w[b] * torch.exp(pred_w[i, j, b]) * 32
                box_h[i, j, b] = anchor_h[b] * torch.exp(pred_h[i, j, b]) * 32

                coord_loss[i, j, b] = coord_scale * (
                    (true_x[i, j, b] - pred_x[i, j, b]) ** 2
                    + (true_y[i, j, b] - pred_y[i, j, b]) ** 2
                    + (true_w[i, j, b] - pred_w[i, j, b]) ** 2
                    + (true_h[i, j, b] - pred_h[i, j, b]) ** 2
                )
                confid_loss[i, j, b] = (utils.calc_IOU(pred, truth) - pred_confid) ** 2 + noobj_scale * (
                    0 - pred_confid
                ) ** 2
                class_loss = 0
                loss = (
                    noobj_scale * sum(noobj_loss)
                    + obj_scale * sum(object_loss)
                    + coord_scale * sum(coord_loss)
                    + class_scale * sum(class_loss)
                )


if __name__ == "__main__":
    yolov2_loss()
