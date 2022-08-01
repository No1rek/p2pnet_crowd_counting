import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def make_path_list(images_folder, labels_folder, val_fraction=0):
    """
        Given folders of images and labels makes a list of image-label pairs
    """
    paths_ds = []
    paths_val = []
    
    image_list = os.listdir(images_folder)
    label_list = os.listdir(labels_folder)
    
    for i,l in zip(image_list, label_list):
        ip = os.path.join(images_folder,i)
        lp = os.path.join(labels_folder,l)
        if (np.random.uniform(0,1) >= val_fraction):
            paths_ds.append((ip,lp))
        else: paths_val.append((ip,lp))
    if val_fraction > 0:
        return paths_ds, paths_val
    return paths_ds

def flattern_predictions(fm_coords, fm_logits):
    """
        Flattens model outputs
    """
    bs, n_anchors, h, w = fm_logits.shape
    n_points = n_anchors*h*w
    logits = fm_logits.flatten(0, 3) # [batch_size*n_points]
    coords = torch.stack((
        fm_coords[:, :n_anchors, :, :].permute(0,2,3,1), # h
        fm_coords[:, n_anchors:, :, :].permute(0,2,3,1) # w
    ), dim=4).permute(0, 3, 1, 2, 4).flatten(0, 3) # [batch_size*n_points, 2]
    return (bs, n_anchors, h, w, n_points), coords, logits

def visualize(coords, image, labels, fname):
    """
        Plots image with predictions and saves it
    """
    lh,lw = labels[:, 0].floor(), labels[:, 1].floor()
    ph,pw = coords[:, 0].floor(), coords[:, 1].floor()
    plt.figure()
    plt.imshow(image)
    plt.scatter(lw, lh, s=5, color="green", label="ground truth")
    plt.scatter(pw, ph, s=5, color="red", label="predictions")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig(fname, bbox_inches='tight')



