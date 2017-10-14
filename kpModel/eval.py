#!/usr/bin/env python

import os
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
from PIL import Image
import caffe, cv2
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from scipy import signal


def showImage(im):
    im = np.array(im, dtype=np.uint8)
    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(im)

def pickUpKeypoint(score, idx, i):
    f = np.zeros((52, 52))
    score = score[:,:,0]
    cv2.circle(f, (25, 25), 25, 1, -1)
    score = signal.convolve2d(score, f, boundary='fill', mode='same') / (25*25*3.14)

    score *= 255.0
    cv2.imwrite('score_' + str(idx) + '_' + str(i) + '_Conv.png' , score)


    point  = np.zeros(score.shape)
    x, y = np.unravel_index(np.argmax(score), score.shape)
    cv2.circle(point, (y, x), 25, 1, -1)
    point *= 255.0
    cv2.imwrite('score_' + str(idx) + '_' + str(i) + '_Point.png' , point)

def segmentAndShowImage(net, imname, idx):

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
        score_norm = score_exp / score_sum # W x H x 2
        score_norm = score_norm[:,:,1:2] # W x H x 1

        pickUpKeypoint(score_norm, idx, i)

        score_norm *= 255
        cv2.imwrite('score_' + str(idx) + '_' + str(i) + '.png' , score_norm)

        label_i = label[i] * 255.0  # 1 x W x H
        label_i = label_i.transpose((1, 2, 0))

        cv2.imwrite('score_' + str(idx) + '_' + str(i) + '_GT.png', label_i)



def main():
    caffe.set_mode_gpu()
    net = caffe.Net('train.prototxt', './kp_snapshot/v2_iter_100000.caffemodel', caffe.TEST)


    val = np.loadtxt('./cropTrain/train.txt', dtype=str)
    for i, line in enumerate(val):
        segmentAndShowImage(net, line, i)
        if i == 1:
            break


if __name__ == '__main__':
    main()
    
