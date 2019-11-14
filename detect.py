import os

import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
import datetime
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
from retinanet import RetinaNet
from encoder import DataEncoder
from augmentations import BaseTransform
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

the_classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=300):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor

        img = cv2.imread(img_path)
        Augmentation = BaseTransform(self.img_size,(104, 117, 123))
        img, _, _ = Augmentation(img)

        img = img[:, :, (2, 1, 0)]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255
        img[0, :, :] = img[0, :, :] / 0.229
        img[1, :, :] = img[1, :, :] / 0.224
        img[2, :, :] = img[2, :, :] / 0.225

        return img_path, img

    def __len__(self):
        return len(self.files)

parser = argparse.ArgumentParser(description='RetinaNet Detection')
parser.add_argument('--trained_model', default='weights/ckpt_1744.pth',type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='output/', type=str,help='Dir to save results')
parser.add_argument('--dataset_root', default='data/samples', help='Dataset root directory path')
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=300, help="size of each image dimension")
args = parser.parse_args()


os.makedirs("output", exist_ok=True)

net = RetinaNet()
net.load_state_dict(torch.load(args.trained_model))


dataloader = DataLoader(
        ImageFolder(args.dataset_root, img_size=args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
    )

TIME=0
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    prev_time = time.time()

    if torch.cuda.is_available():
        input_imgs = input_imgs.cuda()

    loc_preds, cls_preds = net(input_imgs)

    # print('Decoding..')
    encoder = DataEncoder()
    # boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
    boxes, labels, scores = encoder.decode(loc_preds.data, cls_preds.data, 0.2, (args.img_size, args.img_size))

    current_time = time.time()
    inference_time = current_time - prev_time

    if batch_i != 0:
        TIME += inference_time

    print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 6)]

    img = Image.open(img_paths[0])
    img = img.resize((300, 300))
    img = np.array(img)
    # print(img.shape)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    boxe = boxes.data
    label = labels.data
    score = scores.data
    # print(boxe, label, score)

    if boxe is not None:

        for i in range(len(boxe)):

            cls = label[i]
            cls_name = the_classes[int(cls)]
            pt = boxe[i]

            pt[0] = pt[0] if pt[0] > 0 else 0
            pt[1] = pt[1] if pt[1] > 0 else 0
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            # print(pt)
            color = colors[int(cls)]
            # Create a Rectangle patch
            bbox = patches.Rectangle(*coords, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                pt[0],
                pt[1],
                s=cls_name,
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

            # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = img_paths[0].split("/")[-1].split(".")[0]
    plt.savefig(f"output/{filename}.jpg", bbox_inches="tight",pad_inches=0.0)
    plt.close()
print("FPS: %s" % (1/(TIME/5)))


