# Original Code: https://github.com/jiwoon-ahn/irn

import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
import random
from misc import imutils

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

cls_labels_dict = np.load('/utilisateurs/lyuzheng/DeepL/WSSS/Pytorch/08/FPR-master_onyx1/voc12/cls_labels.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    elem_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME, decode_int_filename(img_name) + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((N_CAT), np.float32)

    for elem in elem_list:
        cat_name = elem.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    return np.array([cls_labels_dict[int(img_name.replace('_', ''))] for img_name in img_name_list])

def get_img_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_name_list = np.loadtxt(dataset_path, dtype=str)

    return img_name_list


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name_str, 'img': img}

class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):

        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out['label'] = torch.from_numpy(self.label_list[idx])
        return out


class VOC12CLassificationMixupDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, fp_names,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                         resize_long, rescale, img_normal, hor_flip,
                         crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

        self.fp_idx = []
        for cls_idx in range(20):
            temp = []
            for name in fp_names[cls_idx]:
                temp.append(np.where(self.img_name_list==int(name))[0][0])
            self.fp_idx.append(temp)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out['label'] = torch.from_numpy(self.label_list[idx])
        # randomly select a fp img from  label
        mixup_out_cls = random.choice(np.where(self.label_list[idx] == 1)[0])
        mixup_idx = random.choice(self.fp_idx[mixup_out_cls])
        mix_out = super().__getitem__(mixup_idx)
        mix_out['label'] = torch.from_numpy(self.label_list[mixup_idx])

        return out, mix_out



class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 img_normal=TorchvisionNormalize(),
                 scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, voc12_root, img_normal=img_normal)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.voc12_root))

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name_str, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(self.label_list[idx])}
        return out

class VOC12SegmentationDataset(Dataset):

    def __init__(self, img_name_list_path, label_dir, crop_size, voc12_root,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_method = 'random'):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.voc12_root))
        label = imageio.imread(os.path.join(self.label_dir, name_str + '.png'))

        img = np.asarray(img)

        if self.rescale:
            img, label = imutils.random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, label = imutils.random_lr_flip((img, label))

        if self.crop_method == "random":
            img, label = imutils.random_crop((img, label), self.crop_size, (0, 255))
        else:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        img = imutils.HWC_to_CHW(img)

        return {'name': name, 'img': img, 'label': label}

class VOC12AffinityDataset(VOC12SegmentationDataset):
    def __init__(self, img_name_list_path, label_dir, crop_size, voc12_root,
                 indices_from, indices_to,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False, crop_method=None):
        super().__init__(img_name_list_path, label_dir, crop_size, voc12_root, rescale, img_normal, hor_flip, crop_method=crop_method)

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        reduced_label = imutils.pil_rescale(out['label'], 0.25, 0)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced_label)

        return out



############ OpenImages


def load_img_name_list_openimages(dataset_path):

    img_name_list = open(dataset_path).readlines()
    return img_name_list

class OpenImages_ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True, augment=False):
        self.img_name_list = load_img_name_list_openimages(img_name_list_path)
        self.voc12_root = voc12_root
        self.augment = augment
        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch


    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name_str = self.img_name_list[idx]

        img = np.asarray(imageio.imread(os.path.join(self.voc12_root, name_str.split('\n')[0])))
        if len(img.shape) == 2:
            img = np.concatenate([np.expand_dims(img, 2), np.expand_dims(img, 2), np.expand_dims(img, 2)], axis=2)

        if self.augment:
            if self.resize_long:
                img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

            if self.rescale:
                img = imutils.random_scale(img, scale_range=self.rescale, order=3)
        else:
            if self.resize_long:
                img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])


        if self.img_normal:
            img = self.img_normal(img)
        if self.augment:
            if self.hor_flip:
                img = imutils.random_lr_flip(img)

            if self.crop_size:
                if self.crop_method == "random":
                    img = imutils.random_crop(img, self.crop_size, 0)
                else:
                    img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            if self.augment == False:
                if len(img.shape) == 2:
                    return {}
            # print(img.shape, len(img.shape))


            if self.augment == False:
                w, h, _ = img.shape

                if w < 400 and h < 400:
                    return {}
            img = imutils.HWC_to_CHW(img)

        return {'name': name_str, 'img': img}

class OpenImages_ClassificationDataset(OpenImages_ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, augment=False):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method, augment)
        # self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        # out['label'] = torch.from_numpy(self.label_list[idx])

        return out



class VOC12MetaDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, image_id=None, disjoint_prob=1.0):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.total_list = self.img_name_list
        self.total_label_list = load_image_label_list_from_npy(self.img_name_list)
        self.image_id = image_id
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

        self.img_name_list, self.label_list = self.find_disjoint_class_images()
        self.img_name_list_joint, self.label_list_joint = self.find_joint_class_images()

        self.disjoint_prob = disjoint_prob

    def find_disjoint_class_images(self):
        img_index = list(self.img_name_list).index(int(self.image_id.replace('_', '')))
        target_class = self.label_list[img_index]
        target_cls_idx = set(list(np.nonzero(target_class)[0]))
        new_img_list, new_label_list = [], []
        for img, label in zip(self.img_name_list, self.label_list):
            cls_idx = np.nonzero(label)[0]
            before = len(cls_idx)
            if len(list(set(list(cls_idx)).difference(target_cls_idx))) == before:
                new_img_list.append(img)
                new_label_list.append(label)

        return new_img_list, new_label_list

    def find_joint_class_images(self):
        img_index = list(self.total_list).index(int(self.image_id.replace('_', '')))
        target_class = self.total_label_list[img_index]
        target_cls_idx = set(list(np.nonzero(target_class)[0]))
        print('target', target_cls_idx)

        def get_list(margin):
            new_img_list, new_label_list = [], []
            for img, label in zip(self.total_list, self.total_label_list):
                cls_idx = np.nonzero(label)[0]
                # before = len(cls_idx)
                # print(img, set(list(cls_idx)), len(list(target_cls_idx.difference(set(list(cls_idx))))))
                if len(list(target_cls_idx.difference(set(list(cls_idx))))) == margin:
                    new_img_list.append(img)
                    new_label_list.append(label)
            # print("joint", len(new_img_list))
            print(self.image_id, margin, len(new_img_list))

            return new_img_list, new_label_list

        for m in range(1):
            new_img_list, new_label_list = get_list(m)

            if len(new_label_list) > 20:
                break

        return new_img_list, new_label_list


    def __getitem__(self, idx):
        if self.disjoint_prob == 1.0:
            out = super().__getitem__(idx)
            out['label'] = torch.from_numpy(self.label_list[idx])
            return out

        if random.randrange(0, 1) < self.disjoint_prob:
            # print("disjoint")
            img_name_list = self.img_name_list
            label_list = self.label_list
        else:
            # print(len(self.img_name_list_joint), len(self.label_list_joint))
            img_name_list = self.img_name_list_joint
            label_list = self.label_list_joint

        rand_idx = random.choice(range(0, len(img_name_list)))
        # name = random.choice(img_name_list)
        name = self.img_name_list[rand_idx]
        name_str = decode_int_filename(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name_str, 'img': img, 'label': torch.from_numpy(label_list[rand_idx])}
