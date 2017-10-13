#!/usr/bin/env python

import os
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
from PIL import Image
import caffe, cv2
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

def showImage(im):
    im = np.array(im, dtype=np.uint8)
    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(im)

def segmentAndShowImage(net, imname):

    im = cv2.imread(imname)
    cv2.imwrite('x.png', im)

    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()

    score = net.blobs['score_reshape'].data.copy() # 14 x 2 x W x H
    label = net.blobs['label'].data.copy() # 14 x 1 x W x H
    for i in range(14):
        score_i = score[i] # 2 x W x H
        score_i = score_i.transpose((1, 2, 0)) # W x H x 2

        score_exp = np.exp(score_i)
        score_sum = np.sum(score_exp, axis = 2) # W x H
        score_sum = score_sum[:, :, np.newaxis] # W x H x 1
        score_norm = score_exp / score_sum # W x H x 1
        score_norm = score_norm[:,:,1:2]

        score_norm *= 255
        cv2.imwrite('score_' + str(i) + '.png' , score_norm)


        label_i = label[i] * 255.0  # 1 x W x H
        label_i = label_i.transpose((1, 2, 0))

        cv2.imwrite('label_' + str(i) + '.png', label_i)

        '''
        print(im)
        fig,ax = plt.subplots(figsize=(12,12))
        ax.imshow(im) 
        plt.show()
        '''
    exit(0)

    '''
    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(out, interpolation='nearest')
    '''


def main():
    caffe.set_mode_gpu()
    net = caffe.Net('train.prototxt', './kp_snapshot/_iter_130000.caffemodel', caffe.TEST)


    val = np.loadtxt('./cropTrain/train.txt', dtype=str)
    for line in val:
        segmentAndShowImage(net, line)


if __name__ == '__main__':
    main()
    
