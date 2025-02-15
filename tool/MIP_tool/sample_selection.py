import sys
sys.path.append('/utilisateurs/lyuzheng/DeepL/WSSS/Pytorch/irn_SVM_baseline/')

from tool.MIP_tool.seed_generation import seed_generation_function
import numpy as np
import torch
def data_selection(region_prob, n_pos):
    # the list is an array with length/size of fg_region
    region_prob_list = region_prob.reshape(-1)
    region_list = np.arange(region_prob.shape[0] * region_prob.shape[1])
    # np.random.choice returs specific value, not the index
    # but here, the list is just the index, so the return is the index for fg_region in one-D
    inds = np.random.choice(region_list, p=region_prob_list, size=n_pos, replace=False)
    # generate the link between 2D to 1D
    pairs = np.indices(dimensions=(region_prob.shape[1], region_prob.shape[0])).T
    # p_r is N*2, N = H*W, likes, [0 0], [1 0] ... [W, 0], ... [W, H]
    pairs = pairs.reshape(-1, 2)
    # notice here selection[:, 0] is x, [:, 1] is y.
    selections = pairs[inds]


    return selections

def select_with_prob_fg(args, total_region, threshold, n_pos = None):

    # a_i' = (ai-T)/(1-T)

    total_region = (total_region - threshold) / (1 - threshold)
    total_region[total_region<0] = 0

    if total_region.sum() == 0:
        return np.zeros((2,2))

    fg_region_prob = total_region / (total_region.sum())

    pairs = np.indices(dimensions=(fg_region_prob.shape[0], fg_region_prob.shape[1])).T

    x = int(np.count_nonzero(fg_region_prob))

    'randomly select 1500 data from >T region with confidence CAM_score'
    if n_pos == None:
        n_pos = min(args.data_num, x)
    else:
        n_pos = min(n_pos, x)

    selections = data_selection(fg_region_prob, n_pos)

    # scatter x, y
    # selection x, y
    # plt.imshow(total_region)
    # plt.scatter(selections[:, 0], selections[:, 1], s=1, marker='.', color='black')
    # plt.show()
    return selections
def select_with_prob_bg(args, total_region, ignore_pos, n_pos = None):

    total_region[ignore_pos[0, :], ignore_pos[1, :]] = 0
    bg_region_prob = total_region / (total_region.sum() + 1e-10)

    if bg_region_prob.sum() == 0:
        return np.zeros((2,2))
    else:
        x = int(np.count_nonzero(bg_region_prob))
    'randomly select 1500 data from >T region with confidence CAM_score'
    if n_pos == None:
        n_pos = min(args.data_num, x)
    else:
        n_pos = min(n_pos, x)

    selections = data_selection(bg_region_prob, n_pos)

    # scatter x, y
    # selection x, y
    # plt.imshow(total_region)
    # plt.scatter(selections[:, 0], selections[:, 1], s=1, marker='.', color='black')
    # plt.show()

    return selections


def select_BG_total(args, ForeBackActivation):
    'ForeBackActivation here is in [-1, 1]'
    C, H, W = ForeBackActivation.size()

    '1.define BG'
    'the regions all CAM less than args.confBG'
    # Notice that, not necessary to resize to square like (448, 448), just keep H, W
    bg = np.ones((1, H, W), dtype=np.float32) * args.Tbg

    pred_map = np.concatenate([bg, ForeBackActivation.cpu().detach().numpy()], axis=0)  # [21, H, W]
    pred_map = pred_map.argmax(0)
    # pred_map_backup is for bg latter
    pred_map_backup = pred_map.copy()

    pred_map[pred_map >= 1] = -1
    'label is 0, bg region is 1'
    bg_region = pred_map + 1
    ignore_negpos = np.asarray(np.where(bg_region == 0))

    # uniform selection
    # if the number of bg region < data_num, select all.
    bg_num = np.min((int(args.data_num), np.count_nonzero(bg_region)))
    neg_return = select_with_prob_bg(args, bg_region, ignore_negpos, bg_num)

    # when use neg_return later, is [:0],[:1]，
    # I put a check function code here if necessary.
    # check_selection_utilization(neg_return, H, W)

    # neg_return is size(args.data_num, 2),
    return pred_map_backup, neg_return

def return_pos_from_CAM(args, ForeBackActivation, class_label, img_name):
    C, H, W = ForeBackActivation.size()
    pred_map_backup, neg_selections = select_BG_total(args, ForeBackActivation)

    # Get bg position info, x is 0, y is 1
    minus_x_pos = torch.from_numpy(neg_selections[:, 0])
    minus_y_pos = torch.from_numpy(neg_selections[:, 1])

    x_pos_return = torch.tensor([])
    y_pos_return = torch.tensor([])

    # For return, 1. feature, 2. label, 3.position
    total_Label_selected = np.zeros((1, args.data_num))
    # To count the number of foreground class,
    # if iter == 0, concate bg, if not, just concat new class
    iter = 0
    'white means nothing'
    selected_pos_mask = np.ones((H, W)) * 255
    for label_index in class_label[1]:
        # pred_max get 1 for the pos. where label_num is max.
        pred_max = np.zeros((H, W))
        print(label_index)
        label_num = label_index + 1
        pred_max[pred_map_backup == label_num] = 1
        pred_max = torch.from_numpy(pred_max).to('cuda', non_blocking=True)
        # fg_region_act is the activation for class label_num, only when it is max is considered
        certain_class_cam = ForeBackActivation[label_index]
        fg_region_act = torch.mul(pred_max, certain_class_cam)

        # select by Alex method
        # 1. select foreground, which larger than 0.2
        fg_region_act = fg_region_act.cpu().detach().numpy()
        # fg_region_act = cv2.resize(fg_region_act, (448, 448), interpolation=cv2.INTER_NEAREST)
        fg_threshold = args.Tfg
        pos_return = select_with_prob_fg(args, fg_region_act, fg_threshold, n_pos=None)



        # just in case no fg is selected
        if sum(pos_return).all() <= 0 or sum(neg_selections).all() <= 0:
            break
            # continue

        plus_x_pos = torch.from_numpy(pos_return[:, 0])
        plus_y_pos = torch.from_numpy(pos_return[:, 1])

        selected_pos_mask[plus_y_pos.cpu().detach().numpy(), plus_x_pos.cpu().detach().numpy()] = label_index + 1
        selected_pos_mask[minus_y_pos.cpu().detach().numpy(), minus_x_pos.cpu().detach().numpy()] = 0


        if iter == 0:
            x_pos = torch.cat((minus_x_pos, plus_x_pos))
            y_pos = torch.cat((minus_y_pos, plus_y_pos))

            x_pos_return = torch.cat((minus_x_pos, plus_x_pos))
            y_pos_return = torch.cat((minus_y_pos, plus_y_pos))


            # Label: C*** 0 ***
            num_pos = len(plus_x_pos)
            num_neg = len(minus_x_pos)
            label_pos = np.ones(num_pos) * (label_index + 1)
            label_neg = np.zeros(num_neg)
            Label_selected = np.concatenate((label_neg, label_pos), axis=0)

            total_Label_selected = Label_selected
        else:
            x_pos = plus_x_pos
            y_pos = plus_y_pos

            x_pos_return = torch.cat((x_pos_return, plus_x_pos))
            y_pos_return = torch.cat((y_pos_return, plus_y_pos))
            'select train data and label'

            # Label: C*** 0 ***
            num_pos = len(plus_x_pos)
            label_pos = np.ones(num_pos) * (label_index + 1)
            Label_selected = label_pos
            total_Label_selected = np.concatenate((total_Label_selected, Label_selected), axis=0)

        iter = iter + 1
    total_Label_selected = np.expand_dims(total_Label_selected, axis=0)
    return total_Label_selected, x_pos_return, y_pos_return

def save_initial_seed(label, ForeBackActivation, args, img_name):

    class_label = np.nonzero(label.cpu().detach().numpy())

    # get 2.5K each class
    Label_selected, x_pos, y_pos = return_pos_from_CAM(args, ForeBackActivation, class_label, img_name)

    if x_pos.numel() == 0:
        return x_pos, x_pos, x_pos, x_pos

    return class_label, Label_selected, x_pos, y_pos
def get_info(HybridFeature_selected, Label_selected, x_pos, y_pos, pos):
    start_bg = np.min(pos[1])
    end_bg = np.max(pos[1])
    feature = HybridFeature_selected[:, start_bg:end_bg+1]
    label_s = Label_selected[:, start_bg:end_bg+1]
    xpos = x_pos[start_bg:end_bg+1]
    ypos = y_pos[start_bg:end_bg+1]
    return feature, label_s, xpos, ypos

def get_att_dis(target, behaviored):
    attention_distribution = []

    for i in range(behaviored.size(0)):
        attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
        attention_distribution.append(attention_score)
    attention_distribution = torch.Tensor(attention_distribution)
    return attention_distribution
    # return attention_distribution / torch.sum(attention_distribution, 0)  # 标准化
def refine_bg_prototype(HybridFeature_selected, Label_selected, x_pos, y_pos):
    # get ori bg info
    bg_pos = np.where(Label_selected == 0)
    bg_feature, bg_label_s, bg_xpos, bg_ypos = get_info(HybridFeature_selected, Label_selected, x_pos, y_pos, bg_pos)
    # get ori fg info
    fg_pos = np.where(Label_selected != 0)
    fg_feature, fg_label_s, fg_xpos, fg_ypos = get_info(HybridFeature_selected, Label_selected, x_pos, y_pos, fg_pos)
    # get bg_feature_prototype
    bg_feature_prototype = np.mean(bg_feature, axis=1)
    # calculate cos-sim
    bg_f_proto = torch.from_numpy(bg_feature_prototype.T)
    HF_fore = torch.from_numpy(fg_feature.T)
    fg_bgp_similarity = get_att_dis(target=bg_f_proto, behaviored=HF_fore).cpu().detach().numpy()
    # only select fore whose cos-sim<0
    fg_feature_r = fg_feature[:, fg_bgp_similarity < 0.0]
    fg_label_s_r = fg_label_s[:, fg_bgp_similarity < 0.0]
    fg_xpos_r = fg_xpos[fg_bgp_similarity < 0.0]
    fg_ypos_r = fg_ypos[fg_bgp_similarity < 0.0]
    # concatenate oribg with refine fg
    HF_refine_bgp = np.concatenate((bg_feature, fg_feature_r), axis=1)
    labelsel_refine_bgp = np.concatenate((bg_label_s, fg_label_s_r), axis=1)
    x_pos_refine_bgp = np.concatenate((bg_xpos, fg_xpos_r), axis=0)
    y_pos_refine_bgp = np.concatenate((bg_ypos, fg_ypos_r), axis=0)

    return HF_refine_bgp, labelsel_refine_bgp, x_pos_refine_bgp, y_pos_refine_bgp

# get refine, by bg prototype
def refine_seed(args, F2Feature, ForeBackActivation, x_pos, y_pos, Label_selected, img_name ):
    # select F2 feature
    HybridFeature_selected = return_feature_from_pos(args, F2Feature[0], ForeBackActivation[0],
                                                     x_pos, y_pos)
    # refine by BG_prototype
    _, labelsel_refine_bgp, x_pos_refine_bgp, y_pos_refine_bgp = refine_bg_prototype(
        HybridFeature_selected, Label_selected, x_pos, y_pos)

    # to get N,1 shape for labelsel_refine_bgp
    labelsel_refine_bgp = labelsel_refine_bgp.flatten()

    # save refined seeds
    _, H, W = ForeBackActivation.size()

    return x_pos_refine_bgp, y_pos_refine_bgp, labelsel_refine_bgp
def return_feature_from_pos(args, HybridFeature, ForeBackActivation, x_pos, y_pos):

    # Hybrid could directly save
    HybridFeature_selected = HybridFeature.cpu().detach().numpy()
    # np is C H W, cv2/np is H W C
    HybridFeature_selected = np.transpose(HybridFeature_selected, (1, 2, 0))
    HybridFeature_selected = HybridFeature_selected[y_pos, x_pos, :]
    HybridFeature_selected = np.transpose(HybridFeature_selected, (1, 0))
    return HybridFeature_selected

def return_feature_with_pos(args, feature, ForeBackActivation, x_pos_refine_bgp, y_pos_refine_bgp, class_label, img_name, labelsel_refine_bgp):
    HybridFeature_selected = return_feature_from_pos(args, feature[0], ForeBackActivation,
                                                     x_pos_refine_bgp, y_pos_refine_bgp)

    return HybridFeature_selected, labelsel_refine_bgp

def sample_selection_function(args, label, F2Feature, highres_cam, img_name):
    # norm fg bg to -1,1
    ForeBackActivation = seed_generation_function(args, highres_cam)

    # T1 samples
    class_label, Label_selected, x_pos, y_pos = save_initial_seed(label, ForeBackActivation, args, img_name)

    if x_pos.numel() == 0:
        return np.array([]), np.array([])

    # T2 label
    # get refine, by bg prototype
    x_pos_refine_bgp, y_pos_refine_bgp, labelsel_refine_bgp = refine_seed(args, F2Feature, ForeBackActivation, x_pos,
                                                                          y_pos, Label_selected, img_name)
    # T2 data
    # get F2 feature with refined spatial index
    HyFeat_refine_bgp, labelsel_refine_bgp = return_feature_with_pos(args, F2Feature, ForeBackActivation, x_pos_refine_bgp, y_pos_refine_bgp, class_label, img_name,
                         labelsel_refine_bgp)

    # 这里返回的是2.5K + 2.5K-
    return HyFeat_refine_bgp, labelsel_refine_bgp
