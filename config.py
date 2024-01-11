import os
import numpy as np
import torch
import random

# Set seed
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE = os.getcwd()
data_config = {
    # put combined LUNA16 .mhd files into one folder
    'data_dir': '/home/sharedata/datasets/luna16/LUNG16/subset3',
    # put lung mask downloaded from LUNA16 to this path
    'lung_mask_dir': '/home/sharedata/datasets/luna16/seg-lungs-LUNA16',
    # directory for putting all preprocessed results for training to this path
    'preprocessed_data_dir': '/nvmedata/ligen/luna16',
    # original annotations csv file path
    'annos_dir': os.path.join(BASE, 'annotations/annotations.csv'),
    'annos_excluded_dir': os.path.join(BASE, 'annotations/annotations_excluded.csv'),
    "seriesuids_dir": os.path.join(BASE, 'split/3_val.csv'),

    'new_annos_dir': os.path.join(BASE, 'annotations/new_annotations.csv'),
    'new_annos_excluded_dir': os.path.join(BASE, 'annotations/new_annotations_excludedd.csv'),
    'split_save_dir': os.path.join(BASE, 'split'),
}


def get_anchors(bases, aspect_ratios):
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
            anchors.append([d, h, w])

    return anchors


bases = [5, 10, 20, 30, 50]
aspect_ratios = [[1, 1, 1]]

net_config = {
    # network configuration
    'anchors': get_anchors(bases, aspect_ratios),
    'roi_names': ['nodule'],
    'pad_value': 170,
    'crop_size': [128, 128, 128],
    'bbox_border': 8,
    'stride': 4,
    'max_stride': 16,
    'num_neg': 800,
    'th_neg': 0.02,
    'th_pos_train': 0.5,
    'th_pos_val': 1,
    'num_hard': 3,
    'bound_size': 12,
    'blacklist': [],
    'num_class': 2,
    'aux_loss': False,

    'augtype': {'flip': True, 'rotate': True, 'scale': True, 'swap': False},
    'r_rand_crop': 0.,
    'pad_value': 170,


    'rpn_train_bg_thresh_high': 0.02,
    'rpn_train_fg_thresh_low': 0.5,
    'rpn_train_nms_num': 300,
    'rpn_train_nms_pre_score_threshold': 0.5,
    'rpn_train_nms_overlap_threshold': 0.1,
    'rpn_test_nms_pre_score_threshold': 0.5,
    'rpn_test_nms_overlap_threshold': 0.1,

    # false positive reduction network configuration
    'num_class': 2,
    'rcnn_crop_size': (7, 7, 7),  # can be set smaller, should not affect much
    'rcnn_train_fg_thresh_low': 0.5,
    'rcnn_train_bg_thresh_high': 0.1,
    'rcnn_train_batch_size': 64,
    'rcnn_train_fg_fraction': 0.5,
    'rcnn_train_nms_pre_score_threshold': 0.5,
    'rcnn_train_nms_overlap_threshold': 0.1,
    'rcnn_test_nms_pre_score_threshold': 0.0,
    'rcnn_test_nms_overlap_threshold': 0.1,

    'box_reg_weight': [1., 1., 1., 1., 1., 1.],

    # nodule-detr config
    'hidden_dim': 64,
    'dropout': 0.1,
    'nheads': 8,
    'dim_feedforward': 256,
    'enc_layers': 6,
    'dec_layers': 6,
    'pre_norm': '',
    'return_intermediate_dec': True,
    'position_embedding': 'sine',
    'num_queries': 512,
}


def lr_schedule(epoch, init_lr=0.01, total=120):
    if epoch <= total * 0.5:
        lr = init_lr
    elif epoch <= total * 0.8:
        lr = 0.1 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr


train_config = {
    'net': 'MainNet',
    'batch_size': 4,

    'lr_schedule': lr_schedule,
    'optimizer': 'SGD',
    'init_lr': 0.01,
    'momentum': 0.9,
    'bn_momentum': 0.1,
    'weight_decay': 1e-4,

    'epochs': 120,
    'epoch_save': 1,
    'epoch_rcnn': 65,
    'num_workers': 8,

    'train_set_list': ['split/9_train.csv'],
    'val_set_list': ['split/9_val.csv'],
    'test_set_name': 'split/9_val.csv',
    'label_types': ['bbox'],
    'DATA_DIR': data_config['preprocessed_data_dir'],
    'ROOT_DIR': os.getcwd(),

}


train_config['RESULTS_DIR'] = os.path.join(train_config['ROOT_DIR'], 'results')
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], 'ticnet/fold9')
# train_config['initial_checkpoint'] = os.path.join(train_config['out_dir'], 'model/120.pth')
train_config['initial_checkpoint'] = None
