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
import os.path
import pickle

cache_fname = "svdc2.cache"

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

    def display(_, R, G, B):
        rgb = (R, G, B)
        res = np.zeros_like(img)
        for i in (0, 1, 2):
            _s = s[i].copy()
            _s[:rgb[i]] = 0
            S = np.mat(np.diag(_s))
            res[...,i] = U[i]*S*V[i]
        return res

    def _display_all(RGB):
        if RGB in cache: return cache[RGB]
        cache[RGB] = display(None, RGB, RGB, RGB)
        return cache[RGB]

    if os.path.isfile(cache_fname):
        cache = pickle.load(open(cache_fname, "rb"))
        n = len(cache) - 1
        img = _display_all(n)
    else:
        U, V, s = preprocess(img)
        n = len(s[0])
        cache = {}
        for i in range(n + 1):
            print("caching", i)
            _display_all(i)
        with open(cache_fname, "wb") as fh:
            pickle.dump(cache, fh)
        print("wrote to:", cache_fname)


    def display_all(_, RGB):
        return _display_all(RGB)

    viewer = ImageViewer(img)
    plugin = Plugin(image_filter = display_all)
    plugin.name = ""
    plugin += Slider('RGB', 0, n, 0, 'int', update_on='move')
    viewer += plugin
    viewer.show()

try:
    __IPYTHON__
except:
    import sys
    if len(sys.argv) > 1:
        print("deleting cache")
        try:
            os.unlink(cache_fname)
        except:
            pass
        go(io.imread(sys.argv[1]))
    else:
        go(None)
