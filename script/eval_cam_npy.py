import argparse
import os
import sys
sys.path.append('/utilisateurs/lyuzheng/DeepL/WSSS/Pytorch/irn_SVM_baseline/')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from misc import pyutils
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

def eval_cam(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    labels = []
    preds = []

    for iter, id in enumerate(dataset.ids):
        cam_file = os.path.join(args.cam_out_dir, id + '.npy')

        ext = os.path.exists(cam_file)
        if ext == False:
            continue



        cam_dict = np.load(cam_file, allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())

        gt = dataset.get_example_by_keys(iter, (1,))[0]
        labels.append(gt.copy())

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print({'iou': iou, 'miou': np.nanmean(iou)})

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--chainer_eval_set", default="train", type=str)
    parser.add_argument("--voc12_root", default="/media/data/lyuzheng/Dataset/Official/voc12/VOCdevkit/VOC2012", type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    parser.add_argument("--cam_out_dir", default="/media/data/lyuzheng/Result_2024/11/1112/cam", type=str)

    parser.add_argument("--cam_eval_thres", default=0.2, type=float)
    args = parser.parse_args()

    eval_cam(args)