from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os.path as osp
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

parser = argparse.ArgumentParser(description='visualize tool')
parser.add_argument('--datadir', default='./visdata',
                    help='visualize data directory')
parser.add_argument('--imgidx', default=3,type = int,
                    help='the imgidx of all imgs')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--x', type=int, default=1077)
parser.add_argument('--y', type=int, default=187)
parser.add_argument('--save', default='distribute.png')
parser.add_argument('--modelname', type=str, default='PSMNetCDN',
        help='model name')
# parser.add_argument('--modelname', type=str, default='PSMNetCDN',
#     help='model name')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#PATH and DIRS
args.datadir = os.path.join(args.datadir,args.modelname)
cost_dir = osp.join(args.datadir,"cost_volumes")
offset_dir = osp.join(args.datadir,"offset_volumes")
imgL_dir = osp.join(args.datadir,"imgs")
cost_path = osp.join(cost_dir,str(args.imgidx) + ".npy")
offset_path = osp.join(offset_dir,str(args.imgidx) + ".npy")
imgL_path = osp.join(imgL_dir,str(args.imgidx) + ".png")

cost = np.load(cost_path)
offset =np.load(offset_path)
imgL = Image.open(imgL_path)
print(cost.shape)
H,W,D = cost.shape

def cost_dis(cv,x,y):
    score = cv[y,x,:]
    score_torch = torch.from_numpy(score)
    score_sm = F.softmax(score_torch)
    score_sm = score_sm.numpy()
    return score_sm

def get_offset_dis(offset,x,y):
    score = offset[y,x,:]
    return score

fig = plt.figure()

ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.imshow(imgL)

def onclick(event):
    if event.xdata != None and event.ydata != None and event.button != 1:
        x_ind = int(round(event.xdata))
        y_ind = int(round(event.ydata))
        print("[{},{}]".format(x_ind,y_ind))
        # x: W ,y: H
        cd_sm = cost_dis(cost,x_ind,y_ind)
        offset_dis = get_offset_dis(offset,x_ind,y_ind)
        # plt.subplot(312)
        # plt.figure(2)
        ax2.clear()
        # ax2 = fig.add_subplot(312)
        ax2.plot(range(D),list(cd_sm),linewidth=1)
        #ax2.hist(list(cd_sm),bins=np.linspace(0,D-1,D))
        # ax2.title('score distribution')
        ax3.clear()
        ax3.plot(range(D),list(offset_dis),linewidth=1)
        # ax3.title('score softmax')
        # ax2.show()
        # plt.subplot(313)
        # plt.figure(3)
        ax1.clear()
        # ax3 = fig.add_subplot(313)
        ax1.imshow(imgL)
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


