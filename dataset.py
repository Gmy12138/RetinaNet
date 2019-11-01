from __future__ import print_function

import os
import sys
import random
import glob
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop
from augmentations import SSDAugmentation,BaseTransform


class ListDataset(data.Dataset):
    def __init__(self, root, train, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()


        self._labpath = sorted(glob.glob("%s/*.*" % self.root))
        self._imgpath = [path.replace("labels", "image").replace(".txt", ".jpg") for path in self._labpath]


    def __getitem__(self, index):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.

        img_path = self._imgpath[index].rstrip()
        fname = img_path.split('/')[-1].split('.')[0]

        # print(img_path)
        img = cv2.imread(img_path)
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        h, w, _ = img.shape

        label_path = self._labpath[index].rstrip()
        # print(label_path)

        targets = np.loadtxt(label_path).reshape(-1, 5)

        targets[:, 1] = (targets[:, 1]) / w
        targets[:, 2] = (targets[:, 2]) / h
        targets[:, 3] = (targets[:, 3]) / w
        targets[:, 4] = (targets[:, 4]) / h

        size = self.input_size


        if self.train:

            Augmentation=SSDAugmentation(size=size)
            img, boxe, labels = Augmentation(img, targets[:, 1:], targets[:, 0])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            img =torch.from_numpy(img).permute(2, 0, 1)
            img = img/255
            _, h1, w1 = img.shape
            img[0, :, :] = img[0, :, :] / 0.229
            img[1, :, :] = img[1, :, :] / 0.224
            img[2, :, :] = img[2, :, :] / 0.225

            boxe[:, 0] = w1 * boxe[:, 0]
            boxe[:, 1] = h1 * boxe[:, 1]
            boxe[:, 2] = w1 * boxe[:, 2]
            boxe[:, 3] = h1 * boxe[:, 3]

        else:

            Augmentation=BaseTransform(size=size)
            img, boxe, labels = Augmentation(img, targets[:, 1:], targets[:, 0])
            img = img[:, :, (2, 1, 0)]
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = img / 255
            _, h1, w1 = img.shape
            img[0, :, :] = img[0, :, :] / 0.229
            img[1, :, :] = img[1, :, :] / 0.224
            img[2, :, :] = img[2, :, :] / 0.225

            boxe[:, 0] = w1 * boxe[:, 0]
            boxe[:, 1] = h1 * boxe[:, 1]
            boxe[:, 2] = w1 * boxe[:, 2]
            boxe[:, 3] = h1 * boxe[:, 3]

        boxes = torch.Tensor(boxe)
        labels = torch.LongTensor(labels)
        # img = self.transform(img)
        return img, boxes, labels,fname

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        fname = [x[3] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        # print(num_imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets),fname

    def __len__(self):
        return len(self._labpath)

if __name__ == "__main__":
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = ListDataset(
        root='/home/ecust/gmy/pytorch-retinanet-master/pytorch-retinanet-master/data/NEU-DET/train/labels',
        train=False, transform=transform, input_size=300)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                                             collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets, name in dataloader:
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        print(name)
        # grid = torchvision.utils.make_grid(images, 1)
        # torchvision.utils.save_image(grid, 'a.jpg')
        break