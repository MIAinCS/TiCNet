import argparse
import logging
import os
import sys
import traceback
import warnings
import numpy as np
import pandas as pd
import torch
import setproctitle
from config import net_config, data_config, train_config
from dataset.bbox_reader import BboxReader
from utils.util import Logger
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation
# from net.main_net import build_model
from model import build_model


this_module = sys.modules[__name__]
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
setproctitle.setproctitle('test')

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="eval",
                    help="you want to test or val")
parser.add_argument("--checkpoint", type=str, default=train_config['initial_checkpoint'],
                    help="path to model checkpoint to be used")
parser.add_argument("--out-dir", type=str, default=train_config['out_dir'],
                    help="path to save the results")
parser.add_argument("--test_set_name", type=str, default=train_config['test_set_name'],
                    help="path to test image list")

def main():
    logging.basicConfig(
        format='[%(levelness)s][%(pastime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()

    assert args.mode == 'eval', '-- Mode %s is not supported. ✘' % args.mode
    data_dir = data_config['preprocessed_data_dir']
    test_set_name = args.test_set_name

    initial_checkpoint = args.checkpoint
    model = build_model(net_config)
    model = model.cuda()

    if initial_checkpoint:
        print('-- Loading model from {}...'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
        # print(net.state_dict())
    else:
        print('-- No model checkpoint file specified. ✘')
        return

    save_dir = os.path.join(args.out_dir, 'res', str(epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'FROC')):
        os.makedirs(os.path.join(save_dir, 'FROC'))
    logfile = os.path.join(args.out_dir, 'log_test.txt')
    sys.stdout = Logger(logfile)

    dataset = BboxReader(data_dir, test_set_name, net_config, mode='eval')
    eval(model, dataset, save_dir)

def eval(net, dataset, save_dir=None):
    net.use_rcnn = True
    net.set_mode('eval')

    print('Total # of eval data {}'.format(len(dataset)))
    for i, (input, truth_bboxes, truth_labels, image) in enumerate(dataset):
        try:
            pid = dataset.filenames[i]

            print('-- Scan #{} pid:{} \n-- Predicting {}...'.format(i, pid, image.shape))

            with torch.no_grad():
                input = input.cuda().unsqueeze(0)
                net.forward(input, truth_bboxes, truth_labels)

            rpns = net.rpn_proposals.cpu().numpy()
            detections = net.detections.cpu().numpy()
            ensembles = net.ensemble_proposals.cpu().numpy()

            print('✓ rpn', rpns.shape)
            print('✓ detection', detections.shape)
            print('✓ ensemble', ensembles.shape)



            if len(rpns):
                rpns = rpns[:, 1:]
                np.save(os.path.join(save_dir, '%s_rpns.npy' % pid), rpns)

            if len(detections):
                detections = detections[:, 1:-1]
                np.save(os.path.join(save_dir, '%s_rcnns.npy' % pid), detections)

            if len(ensembles):
                ensembles = ensembles[:, 1:]
                np.save(os.path.join(save_dir, '%s_ensembles.npy' % pid), ensembles)

            # Clear gpu memory
            del input, truth_bboxes, truth_labels, image
            torch.cuda.empty_cache()

        except Exception as e:
            del input, truth_bboxes, truth_labels, image
            torch.cuda.empty_cache()
            traceback.print_exc()
            return

    # Generate prediction csv for the use of performning FROC analysis
    # Save both rpn and rcnn results
    rpn_res = []
    rcnn_res = []
    ensemble_res = []
    for pid in dataset.filenames:
        if os.path.exists(os.path.join(save_dir, '%s_rpns.npy' % pid)):
            rpns = np.load(os.path.join(save_dir, '%s_rpns.npy' % pid))
            rpns = rpns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rpns))
            rpn_res.append(np.concatenate([names, rpns], axis=1))
            os.remove(os.path.join(save_dir, '%s_rpns.npy' % pid))

        if os.path.exists(os.path.join(save_dir, '%s_rcnns.npy' % pid)):
            rcnns = np.load(os.path.join(save_dir, '%s_rcnns.npy' % pid))
            rcnns = rcnns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rcnns))
            rcnn_res.append(np.concatenate([names, rcnns], axis=1))
            os.remove(os.path.join(save_dir, '%s_rcnns.npy' % pid))

        if os.path.exists(os.path.join(save_dir, '%s_ensembles.npy' % pid)):
            ensembles = np.load(os.path.join(
                save_dir, '%s_ensembles.npy' % pid))
            ensembles = ensembles[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(ensembles))
            ensemble_res.append(np.concatenate([names, ensembles], axis=1))
            os.remove(os.path.join(save_dir, '%s_ensembles.npy' % pid))


    rpn_res = np.concatenate(rpn_res, axis=0)
    rcnn_res = np.concatenate(rcnn_res, axis=0)
    ensemble_res = np.concatenate(ensemble_res, axis=0)
    col_names = ['seriesuid', 'coordX', 'coordY',
                 'coordZ', 'diameter_mm', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(
        eval_dir, 'submission_ensemble.csv')

    df = pd.DataFrame(rpn_res, columns=col_names)
    df.to_csv(rpn_submission_path, index=False)

    df = pd.DataFrame(rcnn_res, columns=col_names)
    df.to_csv(rcnn_submission_path, index=False)

    df = pd.DataFrame(ensemble_res, columns=col_names)
    df.to_csv(ensemble_submission_path, index=False)

    # Start evaluating
    if not os.path.exists(os.path.join(eval_dir, 'rpn')):
        os.makedirs(os.path.join(eval_dir, 'rpn'))
    if not os.path.exists(os.path.join(eval_dir, 'rcnn')):
        os.makedirs(os.path.join(eval_dir, 'rcnn'))
    if not os.path.exists(os.path.join(eval_dir, 'ensemble')):
        os.makedirs(os.path.join(eval_dir, 'ensemble'))

    noduleCADEvaluation('annotations/new_annotations.csv',
                        'annotations/new_annotations_exclude.csv',
                        dataset.set_name, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    noduleCADEvaluation('annotations/new_annotations.csv',
                        'annotations/new_annotations_exclude.csv',
                        dataset.set_name, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    noduleCADEvaluation('annotations/new_annotations.csv',
                        'annotations/new_annotations_exclude.csv',
                        dataset.set_name, ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))



def eval_single(net, input):
    with torch.no_grad():
        input = input.cuda().unsqueeze(0)
        logits = net.forward(input)
        logits = logits[0]

    masks = logits.cpu().data.numpy()
    masks = (masks > 0.5).astype(np.int32)
    return masks


if __name__ == '__main__':
    main()
