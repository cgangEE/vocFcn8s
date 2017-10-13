#!/usr/bin/python

import os
#os.environ['GLOG_minloglevel'] = '1'
import caffe
import surgery, score
import sys
import numpy as np


def getPixNum(solver, blobName):
    shape = solver.net.blobs[blobName].data.shape
    ret = 1
    for i in shape:
        ret *= i
    return ret


def printNetInfo(solver):
    for k, v in solver.net.blobs.items():
        print(k, v.data.shape)
    print('------------------------------------------')
    for k, v in solver.net.params.items():
        print(k, v[0].data.shape)
    print('------------------------------------------')


def trainTattooModel():
    caffe.set_mode_gpu()
    weights = '../caffemodel/iter58000.caffemodel'
#    weights = './fcn8s-heavy-pascal.caffemodel'
    solver = caffe.SGDSolver('solver.prototxt')
    solver.net.copy_from(weights)

    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)

    l = solver.net.blobs['score']

    train = np.loadtxt('./tattoo/train.txt', dtype=str)
    val = np.loadtxt('./tattoo/val.txt', dtype=str)

    for _ in range(1):
        solver.step(len(train))
#        score.seg_tests(solver, val)
#        sys.stdout.flush()

    return solver


if __name__ == '__main__':
    trainTattooModel()

