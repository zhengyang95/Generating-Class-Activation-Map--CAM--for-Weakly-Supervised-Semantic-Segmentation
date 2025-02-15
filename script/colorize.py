# -*- encoding: utf-8 -*-
'''
@File    :   Save_Feature.py
@Modify Time      @Author
------------      --------
18/0/2023 16:03   zhengyang

@Desciption
'''
import matplotlib
import os
from PIL import Image
from pathlib import Path
import numpy as np
import sys
import cv2

import torch
import matplotlib.pyplot as plt

'保存dark mask'
def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])
def get_pascal_labels_unc():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128],[255,255,255]])

def decode_segmap_uncertain(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    n_classes = 22
    label_colours = get_pascal_labels_unc()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    n_classes = 21
    label_colours = get_pascal_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def decode_seg_map_sequence(label_masks):
    if label_masks.ndim == 2:
        label_masks = label_masks[None, :, :]

    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap_uncertain(label_mask)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def cv2_dark_mask_save(args, img_name, pred_map):
    dark_mask_path = args.CAM_mask_data_dir + 'dark_mask/'
    if not os.path.exists(dark_mask_path):
        os.makedirs(dark_mask_path)

    pred_save_path = dark_mask_path + img_name + '.png'
    cv2.imwrite(pred_save_path, pred_map)
    return dark_mask_path

def convert_gray_color(aug_grey, aug_color):
    if not os.path.exists(aug_color):
        os.makedirs(aug_color)

    paths = list(Path(aug_grey).glob('*'))
    'read in grey'
    for i in paths:
        fields = i.name.strip().split()
        img_id = fields[0][:11]
        grey_path = aug_grey + img_id + '.png'

        grey_img = Image.open(grey_path)

        color_img = decode_seg_map_sequence(np.array(grey_img)) * 255

        color_img = color_img[0].detach().numpy()
        color_img = color_img.astype(np.float32) / 255.
        color_img = np.transpose(color_img, (1, 2, 0))

        'save in color'
        color_path = aug_color + img_id + '.png'
        matplotlib.image.imsave(color_path, color_img)

def convert_gray_color_bylist(aug_grey, aug_color,list):
    if not os.path.exists(aug_color):
        os.makedirs(aug_color)
    # Open file
    fileHandler = open(list, "r")
    # Get list of all lines in file
    listOfLines = fileHandler.readlines()
    # Close file
    fileHandler.close()

    'read in grey'
    for img_id in listOfLines:

        img_id = img_id[:11]
        grey_path = aug_grey + img_id + '.png'
        if not os.path.exists(grey_path):
            continue
        grey_img = Image.open(grey_path)
        # gt_map = torch.from_numpy(np.array(gt_map))

        color_img = decode_seg_map_sequence(np.array(grey_img)) * 255

        color_img = color_img[0].detach().numpy()
        color_img = color_img.astype(np.float32) / 255.
        color_img = np.transpose(color_img, (1, 2, 0))

        'save in color'
        color_path = aug_color + img_id + '_pred.png'
        matplotlib.image.imsave(color_path, color_img)


if __name__ == '__main__':
    dark_mask_path = '/media/data/lyuzheng/Result_2024/11/1112/cam/png/'
    color_mask_path = os.path.join(dark_mask_path, 'color/')
    os.makedirs(color_mask_path, exist_ok=True)
    list ='/utilisateurs/lyuzheng/DeepL/2024_11_submit/ResNet_50_Classification/voc12/train.txt'

    convert_gray_color_bylist(dark_mask_path, color_mask_path, list)