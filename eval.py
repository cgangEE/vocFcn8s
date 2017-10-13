#!/usr/bin/python
import os
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
from PIL import Image
import caffe
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

def showImage(im):
    im = np.array(im, dtype=np.uint8)
    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(im)

def segmentAndShowImage(imageName, net):
    imageName = './tattoo' + imageName[1:]
    im = Image.open(imageName)
    showImage(im)

    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)

    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(out, interpolation='nearest')


def main():
    caffe.set_mode_gpu()
    net = caffe.Net('cgDeploy.prototxt', 'iter8000.caffemodel', caffe.TEST)

    val = np.loadtxt('./tattoo/val.txt', dtype=str)
    for idx, line in enumerate(val):
        segmentAndShowImage(line, net)
        if idx == 10:
            break
    


if __name__ == '__main__':
    main()
    plt.show()
    
