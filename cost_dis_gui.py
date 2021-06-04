from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
#if you use pycharm,need these two
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
# import matplotlib
from matplotlib.widgets import Slider,RadioButtons
from PIL import Image

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--load', default='000168_10.tar',
                    help='loading model')
parser.add_argument('--loadimg', default='000168_10.png',
                    help='loading img')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--x', type=int, default=1077)
parser.add_argument('--y', type=int, default=187)
parser.add_argument('--save', default='distribute.png')




args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

cv = torch.load(args.load)['cv']
print(cv.shape)
D,H,W = cv.shape


def cost_dis(cv,x,y):
    score = cv[:,y,x]
    # score = score - score.min()
    score_torch = torch.from_numpy(score)
    score_sm = F.softmax(score_torch)
    score_sm = score_sm.numpy()
    return score, score_sm



# if __name__ == '__main__':
#     cd = cost_dis(cv,args.x,args.y)
#     plt.plot(range(D),list(cd),linewidth=1)
#     plt.show()

img = Image.open(args.loadimg)
fig = plt.figure()

ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.imshow(img)

def onclick(event):
    if event.xdata != None and event.ydata != None and event.button != 1:
        print("123")
        x_ind = int(round(event.xdata))
        y_ind = int(round(event.ydata))
        print("x is {}, y is {}".format(x_ind,y_ind))
        cd, cd_sm = cost_dis(cv,x_ind,y_ind)
        # plt.subplot(312)
        # plt.figure(2)
        ax2.clear()
        # ax2 = fig.add_subplot(312)
        ax2.plot(range(D),list(cd),linewidth=1)
        # ax2.title('score distribution')
        ax3.clear()
        ax3.plot(range(D),list(cd_sm),linewidth=1)
        # ax3.title('score softmax')
        # ax2.show()
        # plt.subplot(313)
        # plt.figure(3)
        ax1.clear()
        # ax3 = fig.add_subplot(313)
        ax1.imshow(img)
        ax1.plot(x_ind,y_ind,'yo')
        # ax3.show()
        fig.show()

cid = fig.canvas.mpl_connect('button_press_event',onclick)

plt.show()

# plt.figure(1)
# plt.imshow(img)
# pos = plt.ginput(1)
# # print(pos)
# # print(pos[0][0])
# # print(pos[0][1])

# x_ind = int(round(pos[0][0]))
# y_ind = int(round(pos[0][1]))
# cd = cost_dis(cv,x_ind,y_ind)
# plt.figure(2)
# plt.plot(range(D),list(cd),linewidth=1)
# plt.show()


