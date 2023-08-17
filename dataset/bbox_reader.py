import numpy as np
import torch
from torch.utils.data import Dataset
import os
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
import math
import time
import nrrd


class BboxReader(Dataset):
    def __init__(self, data_dir, set_name, cfg, mode='train', split_combiner=None):
        self.mode = mode
        self.cfg = cfg
        self.r_rand = cfg['r_rand_crop']
        self.augtype = cfg['augtype']
        self.pad_value = cfg['pad_value']
        self.data_dir = data_dir
        self.stride = cfg['stride']
        self.blacklist = cfg['blacklist']
        self.blacklist = []
        self.set_name = set_name

        labels = []
        if set_name.endswith('.csv'):
            self.filenames = np.genfromtxt(set_name, dtype=str)
        elif set_name.endswith('.npy'):
            self.filenames = np.load(set_name)

        if mode != 'test':
            self.filenames = [f for f in self.filenames if (f not in self.blacklist)]

        for fn in self.filenames:
            l = np.load(os.path.join(data_dir, '%s_bboxes.npy' % fn))
            if np.all(l == 0):
                l = np.array([])
            labels.append(l)

        self.sample_bboxes = labels
        if self.mode in ['train', 'val', 'eval']:
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0:
                    for t in l:
                        self.bboxes.append([np.concatenate([[i], t])])
            self.bboxes = np.concatenate(self.bboxes, axis=0).astype(np.float32)
        self.crop = Crop(cfg)
        self.split_combiner = split_combiner

    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
        is_random_img = False
        if self.mode in ['train', 'val']:
            if idx >= len(self.bboxes):
                is_random_crop = True
                idx = idx % len(self.bboxes)
                is_random_img = np.random.randint(2)
            else:
                is_random_crop = False
        else:
            is_random_crop = False

        if self.mode in ['train', 'val']:
            if not is_random_img:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = self.load_img(filename)
                bboxes = self.sample_bboxes[int(bbox[0])]

                isScale = self.augtype['scale'] and (self.mode == 'train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, is_random_crop)
                if self.mode == 'train' and not is_random_crop:
                    sample, target, bboxes = augment(sample, target, bboxes, do_flip=self.augtype['flip'],
                                                     do_rotate=self.augtype['rotate'], do_swap=self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.filenames))
                filename = self.filenames[randimid]
                imgs = self.load_img(filename)
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype['scale'] and (self.mode == 'train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)

            if sample.shape[1] != self.cfg['crop_size'][0] or sample.shape[2] != \
                    self.cfg['crop_size'][1] or sample.shape[3] != self.cfg['crop_size'][2]:
                print(filename, sample.shape)

            sample = (sample.astype(np.float32) - 128) / 128
            bboxes = fillter_box(bboxes, self.cfg['crop_size'])
            label = np.ones(len(bboxes), dtype=np.int32)
            bboxes[:, -1] = bboxes[:, -1] + self.cfg['bbox_border']
            bboxes = np.concatenate((bboxes, bboxes[:, -1][..., np.newaxis], bboxes[:, -1][..., np.newaxis]), axis=1)

            return [torch.from_numpy(sample), bboxes, label]

        if self.mode in ['eval']:
            image = self.load_seg_img(self.filenames[idx])

            original_image = image[0]

            image = pad2factor(image[0])
            image = np.expand_dims(image, 0)

            truth_bboxes = None
            truth_labels = None

            input = (image.astype(np.float32) - 128.) / 128.

            return [torch.from_numpy(input).float(), truth_bboxes, truth_labels, original_image]

    def __len__(self):
        if self.mode == 'train':
            return int(len(self.bboxes) / (1 - self.r_rand))
        elif self.mode == 'val':
            return len(self.bboxes)
        else:
            return len(self.filenames)

    def load_img(self, path_to_img):
        img, _ = nrrd.read(os.path.join(self.data_dir, '%s_seg.nrrd' % (path_to_img)))
        img = img[np.newaxis, ...]
        return img

    def load_seg_img(self, path_to_img):
        img, _ = nrrd.read(os.path.join(self.data_dir, '%s_seg.nrrd' % (path_to_img)))
        img = img[np.newaxis, ...]
        return img


def pad2factor(image, factor=16, pad_value=0):
    depth, height, width = image.shape
    d = int(math.ceil(depth / float(factor))) * factor
    h = int(math.ceil(height / float(factor))) * factor
    w = int(math.ceil(width / float(factor))) * factor

    pad = [[0, d - depth], [0, h - height], [0, w - width]]

    image = np.pad(image, pad, 'constant', constant_values=pad_value)

    return image


def fillter_box(bboxes, size):
    res = []
    for box in bboxes:
        if np.all(box[:3] - box[-1] / 2 > 0) and np.all(box[:3] + box[-1] / 2 < size):
            res.append(box)
    return np.array(res)


def augment(sample, target, bboxes, do_flip=True, do_rotate=True, do_swap=True):
    #                     angle1 = np.random.rand()*180
    if do_rotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand() * 180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if do_swap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]

    if do_flip:
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]

    return sample, target, bboxes


class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False):
        if isScale:
            radiusLim = [8., 120.]
            scaleLim = [0.75, 1.25]
            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1])
                , np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)

        start = []
        for i in range(3):
            # start.append(int(target[i] - crop_size[i] / 2))
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
            else:
                s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])
                e = np.min([crop_size[i] / 2, imgs.shape[i + 1] / 2 - bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            if s > e:
                start.append(np.random.randint(e, s))  # !
            else:
                start.append(int(target[i]) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))

        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
        xx, yy, zz = np.meshgrid(
            np.linspace(normstart[0], normstart[0] + normsize[0], int(self.crop_size[0] / self.stride)),
            np.linspace(normstart[1], normstart[1] + normsize[1], int(self.crop_size[1] / self.stride)),
            np.linspace(normstart[2], normstart[2] + normsize[2], int(self.crop_size[2] / self.stride)), indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

        pad = [[0, 0]]
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:,
               int(max(start[0], 0)):int(min(start[0] + crop_size[0], imgs.shape[1])),
               int(max(start[1], 0)):int(min(start[1] + crop_size[1], imgs.shape[2])),
               int(max(start[2], 0)):int(min(start[2] + crop_size[2], imgs.shape[3]))]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]

        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j] * scale
        return crop, target, bboxes, coord
