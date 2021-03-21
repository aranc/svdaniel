#!/usr/bin/python
from __future__ import division
import scipy.linalg
import numpy as np
import skimage
from skimage import data, io
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.plugins import Plugin
from scipy.linalg import svd

def go(img):
    def preprocess(img):
        U = {}
        V = {}
        s = {}
        for i in (0, 1, 2):
            res = svd(img[...,i], full_matrices=False)
            U[i] = np.mat(res[0])
            V[i] = np.mat(res[2])
            s[i] = res[1]
        return U, V, s
    U, V, s = preprocess(img)

    def display(_, _from, _to):
        res = np.zeros_like(img)
        for i in (0, 1, 2):
            _s = s[i].copy()
            _s[:_from] = 0
            _s[_to:] = 0
            S = np.mat(np.diag(_s))
            res[...,i] = U[i]*S*V[i]
        return res

    viewer = ImageViewer(img)
    plugin = Plugin(image_filter = display)
    plugin.name = ""
    plugin += Slider('_from', 0, len(s[0]), 0, 'int', update_on='move')
    plugin += Slider('_to', 0, len(s[0]), len(s[0]), 'int', update_on='move')
    viewer += plugin
    viewer.show()

try:
    __IPYTHON__
except:
    import sys
    go(io.imread(sys.argv[1]))
