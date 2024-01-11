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
from net.main_net import build_model



this_module = sys.modules[__name__]
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
setproctitle.setproctitle('ticnet-test')

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="eval",
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default=train_config['initial_checkpoint'],
                    help="path to model weights to be used")
parser.add_argument("--out-dir", type=str, default=train_config['out_dir'],
                    help="path to save the results")
parser.add_argument("--test_set_name", type=str, default=train_config['test_set_name'],
                    help="path to test image list")

def main():
    logging.basicConfig(
        format='[%(levelness)s][%(pastime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()

    assert args.mode == 'eval', 'Mode %s is not supported. ✘' % args.mode
    data_dir = data_config['preprocessed_data_dir']
    test_set_name = args.test_set_name

    initial_checkpoint = args.weight
    model = build_model(net_config)
    model = model.cuda()

    if initial_checkpoint:
        print(f'Loading model from {initial_checkpoint}...')
        checkpoint = torch.load(initial_checkpoint)
        epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])

    else:
        print('No model weight file specified.')
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
    rpn_res = []
    rcnn_res = []
    ensemble_res = []

    print(f'Total num of eval data {len(dataset)}')
    for i, (input, truth_bboxes, truth_labels, image) in enumerate(dataset):
        try:
            pid = dataset.filenames[i]

            print(f'Scan {i} pid {pid} shape {image.shape}')

            with torch.no_grad():
                input = input.cuda().unsqueeze(0)
                net.forward(input, truth_bboxes, truth_labels)

            rpns = net.rpn_proposals.cpu().numpy()
            rcnns = net.detections.cpu().numpy()
            ensembles = net.ensemble_proposals.cpu().numpy()
            keeps = []
            for index in range(len(ensembles)):
                if ensembles[index][1] > 0.5:
                    keeps.append(ensembles[index])
            keeps = np.array(keeps)

            print(f'rpn: {rpns.shape}, rcnn: {rcnns.shape}, ensemble: {ensembles.shape}')
 

            if len(rpns):
                rpns = rpns[:, 1:]
                rpns = rpns[:, [3, 2, 1, 4, 0]]
                names = np.array([[pid]] * len(rpns))
                rpn_res.append(np.concatenate([names, rpns], axis=1))

            if len(rcnns):
                rcnns = rcnns[:, 1:-1]
                rcnns = rcnns[:, [3, 2, 1, 4, 0]]
                names = np.array([[pid]] * len(rcnns))
                rcnn_res.append(np.concatenate([names, rcnns], axis=1))

            if len(ensembles):
                ensembles = ensembles[:, 1:]
                ensembles = ensembles[:, [3, 2, 1, 4, 0]]
                names = np.array([[pid]] * len(ensembles))
                ensemble_res.append(np.concatenate([names, ensembles], axis=1))      

        except Exception as e:
            traceback.print_exc()
        finally:
            # Clear gpu memory
            del input, truth_bboxes, truth_labels, image
            torch.cuda.empty_cache()

    # Generate prediction csv for the use of performning FROC analysis
    # Save both rpn and rcnn results
    rpn_res = np.concatenate(rpn_res, axis=0)
    rcnn_res = np.concatenate(rcnn_res, axis=0)
    ensemble_res = np.concatenate(ensemble_res, axis=0)

    col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')
   

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
                        'annotations/new_annotations_excluded.csv',
                        dataset.set_name, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    noduleCADEvaluation('annotations/new_annotations.csv',
                        'annotations/new_annotations_excluded.csv',
                        dataset.set_name, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    noduleCADEvaluation('annotations/new_annotations.csv',
                        'annotations/new_annotations_excluded.csv',
                        dataset.set_name, ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))



if __name__ == '__main__':
    main()
