import argparse
import os
import pprint
import sys
import time
import traceback
import warnings
import setproctitle
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import data_config, train_config, net_config
from dataset.bbox_reader import BboxReader
from dataset.collate import train_collate
from utils.util import Logger
from net.main_net import build_model


warnings.filterwarnings("ignore")
# setup cuda device
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
this_module = sys.modules[__name__]
setproctitle.setproctitle("ticnet-train")

parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument('--batch-size', default=train_config['batch_size'], type=int, metavar='N',
                    help='batch size')
parser.add_argument('--epochs', default=train_config['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-rcnn', default=train_config['epoch_rcnn'], type=int, metavar='NR',
                    help='number of epochs before training rcnn')
parser.add_argument('--epoch-save', default=train_config['epoch_save'], type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--checkpoint', default=train_config['initial_checkpoint'], type=str, metavar='pth',
                    help='checkpoint to use')
parser.add_argument('--optimizer', default=train_config['optimizer'], type=str, metavar='SPLIT',
                    help='which split set to use')
parser.add_argument('--init-lr', default=train_config['init_lr'], type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', default=train_config['weight_decay'], type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--out-dir', default=train_config['out_dir'], type=str, metavar='OUT',
                    help='directory to save results of this training')
parser.add_argument('--train-set-list', default=train_config['train_set_list'], nargs='+', type=str,
                    help='train set paths list')
parser.add_argument('--val-set-list', default=train_config['val_set_list'], nargs='+', type=str,
                    help='val set paths list')
parser.add_argument('--data-dir', default=train_config['DATA_DIR'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--num-workers', default=train_config['num_workers'], type=int, metavar='N',
                    help='number of data loading workers')


def main():
    # Load training configuration
    args = parser.parse_args()
    lr_schedule = train_config['lr_schedule']
    label_types = train_config['label_types']

    train_dataset_list = []
    val_dataset_list = []
    for i in range(len(args.train_set_list)):
        set_name = args.train_set_list[i]
        label_type = label_types[i]

        assert label_type == 'bbox', 'DataLoader not support'
        dataset = BboxReader(args.data_dir, set_name, net_config, mode='train')
        train_dataset_list.append(dataset)

    for i in range(len(args.val_set_list)):
        set_name = args.val_set_list[i]
        label_type = label_types[i]

        assert label_type == 'bbox', 'DataLoader not support'
        dataset = BboxReader(args.data_dir, set_name, net_config, mode='val')
        val_dataset_list.append(dataset)

    train_loader = DataLoader(ConcatDataset(train_dataset_list), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate, drop_last=True)
    val_loader = DataLoader(ConcatDataset(val_dataset_list), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate, drop_last=True)


    # Initialize network
    model = build_model(net_config)
    model = model.cuda()

    optimizer = getattr(torch.optim, args.optimizer)
    # SGD
    optimizer = optimizer(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

    start_epoch = 0

    if args.checkpoint:
        print(f'Loading model from {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch']
        state = model.state_dict()
        state.update(checkpoint['state_dict'])

        try:
            model.load_state_dict(state)
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print('Load checkpoint failed.')
            traceback.print_exc()

    start_epoch = start_epoch + 1

    model_out_dir = os.path.join(args.out_dir, 'model')
    tb_out_dir = os.path.join(args.out_dir, 'runs')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    logfile = os.path.join(args.out_dir, 'log_train.txt')
    sys.stdout = Logger(logfile)

    print('[Training configuration]')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('[Model configuration]')
    pprint.pprint(net_config)

    print(f'Start_epoch {start_epoch}, out_dir {args.out_dir}')
    print(f'Length of train loader {len(train_loader)}, length of valid loader {len(val_loader)}')

    # Write graph to tensorboard for visualization
    writer = SummaryWriter(tb_out_dir)
    train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))

    for i in tqdm(range(start_epoch, args.epochs + 1), desc='Total', ncols=100):
        # learning rate schedule
        if isinstance(optimizer, torch.optim.SGD):
            lr = lr_schedule(i, init_lr=args.init_lr, total=args.epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = 1e-3

        if i >= args.epoch_rcnn:
            model.use_rcnn = True
        else:
            model.use_rcnn = False

        start = time.time()
        batch_size = train_config['batch_size']

        print('\n')
        print(f'Start epoch {i}, batch_size {batch_size}, lr {lr}, use_rcnn {model.use_rcnn}')

        train(model, train_loader, optimizer, i, train_writer)
        validate(model, val_loader, i, val_writer)

        end = time.time()
        print(f'Finish Epoch {i}, Running time {int(end - start)}s\n')

        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        if i % args.epoch_save == 0:
            torch.save({
                'epoch': i,
                'out_dir': args.out_dir,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict()},
                os.path.join(model_out_dir, '%03d.pth' % i))

    writer.close()
    train_writer.close()
    val_writer.close()


def train(net, train_loader, optimizer, epoch, writer):
    net.set_mode('train')
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []

    total_loss = []

    with tqdm(enumerate(train_loader), total=len(train_loader), desc='[Train %d]' % epoch, ncols=100) as t:
        try:
            for j, (input, truth_box, truth_label) in t:
                input = Variable(input).cuda()
                truth_box = np.array(truth_box)
                truth_label = np.array(truth_label)

                net(input, truth_box, truth_label)

                loss = net.loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
                rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
                rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
                rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())
                

                total_loss.append(loss.cpu().data.item())

        except KeyboardInterrupt:
            raise
        finally:
            t.close()

    print('\n')
    print(f'Train Epoch {epoch}, iter {j}, loss {np.average(total_loss)}')
    print(f'rpn_cls {np.average(rpn_cls_loss)}, rpn_reg {np.average(rpn_reg_loss)}, \
          rcnn_cls {np.average(rcnn_cls_loss)}, rcnn_reg {np.average(rcnn_reg_loss)}')

    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)

    del input, truth_box, truth_label
    del net.rpn_proposals, net.detections
    del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss
    del net.rpn_logits_flat, net.rpn_deltas_flat

    if net.use_rcnn:
        del net.rcnn_logits, net.rcnn_deltas

    torch.cuda.empty_cache()


def validate(net, val_loader, epoch, writer):
    net.set_mode('valid')
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []

    total_loss = []

    with tqdm(enumerate(val_loader), total=len(val_loader), desc='[Val %d]' % epoch, ncols=100) as t:
        try:
            for j, (input, truth_box, truth_label) in t:
                with torch.no_grad():
                    input = Variable(input).cuda()
                    truth_box = np.array(truth_box)
                    truth_label = np.array(truth_label)

                    net(input, truth_box, truth_label)
                    loss = net.loss()
            t.close()

        except KeyboardInterrupt:
            t.close()
            raise

        rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())

        total_loss.append(loss.cpu().data.item())

    print('\n')
    print(f'Validate Epoch {epoch}, iter {j}, loss {np.average(total_loss)}')
    print(f'rpn_cls {np.average(rpn_cls_loss)}, rpn_reg {np.average(rpn_reg_loss)}, \
          rcnn_cls {np.average(rcnn_cls_loss)}, rcnn_reg {np.average(rcnn_reg_loss)}')

    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)
 

    del input, truth_box, truth_label
    del net.rpn_proposals, net.detections
    del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss
    del net.rpn_logits_flat, net.rpn_deltas_flat

    if net.use_rcnn:
        del net.rcnn_logits, net.rcnn_deltas

    torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
