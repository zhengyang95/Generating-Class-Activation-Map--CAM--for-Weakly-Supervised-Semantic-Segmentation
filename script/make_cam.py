"在make_cam基础上改的"
import os
import sys
sys.path.append('/utilisateurs/lyuzheng/DeepL/WSSS/Pytorch/irn_SVM_baseline/')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader,Subset
import torch.nn.functional as F
from torch.backends import cudnn
import cv2
import numpy as np
import importlib
import argparse
import voc12.dataloader
from misc import torchutils, imutils

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        data_set_length = len(data_loader.dataset)
        indices = list(range(0, data_set_length))
        subset = Subset(data_loader.dataset, indices)
        new_data_loader = DataLoader(subset, shuffle=False)
        for iter, pack in enumerate(new_data_loader, start=0):

            img_name = pack['name'][0]

            label = pack['label'][0]
            size = pack['size']
            img_ori = pack['img'][0]

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            _, _, _, H_img, W_img = img_ori.shape


            outputs = [model(img[0].cuda(non_blocking=True), "F2")
                       for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(o[0], strided_size, mode='bilinear', align_corners=False) for o
                 in outputs]), 0)[0]

            highres_cam = [F.interpolate(torch.unsqueeze(torch.squeeze(o[0]), 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam = strided_cam / (F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5)


            highres_cam = highres_cam[valid_cat]
            highres_cam = highres_cam / (F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5)

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})


            cams = np.pad(highres_cam.cpu().detach(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            keys = np.pad(valid_cat.cpu().detach() + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            os.makedirs(os.path.join(args.cam_out_dir, 'png'), exist_ok=True)
            cv2.imwrite(os.path.join(args.cam_out_dir, 'png', img_name + '.png'), cls_labels)

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')

def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + 'best.pth'), strict=True)
    # model.load_state_dict(torch.load('/media/data/lyuzheng/Pretrained' + 'res50_cam_orig.pth'), strict=True)
    model.eval()


    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--voc12_root", default="/media/data/lyuzheng/Dataset/Official/voc12/VOCdevkit/VOC2012",
                        type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list",
                        default="/utilisateurs/lyuzheng/DeepL/2024_11_submit/ResNet_50_Classification/voc12/train_aug.txt",
                        type=str)
    parser.add_argument("--val_list",
                        default="/utilisateurs/lyuzheng/DeepL/2024_11_submit/ResNet_50_Classification/voc12/train.txt",
                        type=str)
    parser.add_argument("--infer_list",
                        default="/utilisateurs/lyuzheng/DeepL/2024_11_submit/ResNet_50_Classification/voc12/train.txt",
                        type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam_multi_feature", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=30, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.2, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name",
                        default="/media/data/lyuzheng/Result_2024/10/1012_ori_MIP_CAM/cls_weight/", type=str)
    parser.add_argument("--cam_out_dir", default="/media/data/lyuzheng/Result_2024/11/1112/cam",
                        type=str)
    # Step
    parser.add_argument("--train_cam_pass", default=False)
    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--eval_cam_pass", default=True)

    args = parser.parse_args()

    os.makedirs(args.cam_weights_name, exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    # pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    run(args)
