  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from PIL import Image
import sys
import os
import json
import torch
from glob import glob
import logging






def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines




def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img


def read_disp(filename, subset=False):
    # Scene Flow dataset
    if filename.endswith('pfm'):
        # For finalpass and cleanpass, gt disparity is positive, subset is negative
        disp = np.ascontiguousarray(_read_pfm(filename)[0])
        if subset:
            disp = -disp
    # KITTI
    elif filename.endswith('png'):
        disp = _read_kitti_disp(filename)
    elif filename.endswith('npy'):
        disp = np.load(filename)
    else:
        raise Exception('Invalid disparity file format!')
    return disp  # [H, W]


def _read_kitti_disp(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth


def get_depth(depth_file):
    return np.load(depth_file).astype(np.float32)