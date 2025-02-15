import argparse
import os
import sys
sys.path.append('/utilisateurs/lyuzheng/DeepL/WSSS/Pytorch/irn_SVM_baseline/')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from misc import pyutils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--voc12_root", default="/media/data/lyuzheng/Dataset/Official/voc12/VOCdevkit/VOC2012", type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="train_aug.txt", type=str)
    parser.add_argument("--val_list", default="val.txt", type=str)
    parser.add_argument("--infer_list", default="train.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam_multi_feature", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.20, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")



    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="/media/data/lyuzheng/Result_2024/10/1012_ori_MIP_CAM/cls_weight/best", type=str)
    parser.add_argument("--cam_out_dir", default="/media/data/lyuzheng/Result_2024/10/1012_ori_MIP_CAM/cam", type=str)


    # Step
    parser.add_argument("--train_cam_pass", default=True)
    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--eval_cam_pass", default=True)

    args = parser.parse_args()

    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.cam_weights_name, exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)




