#!/usr/bin/env python

import os, cPickle
from matplotlib import pyplot as plt
os.environ['GLOG_minloglevel'] = '2'
plt.rcParams.update({'figure.max_open_warning': 0})

import numpy as np
from PIL import Image
import caffe, cv2, json, cPickle
from scipy import signal
from timer import Timer


def getKeypoint(imname, boxes, box_thresh, net, t, idx):

    image = dict()
    image['image_id'] = imname[:-4]

    im = cv2.imread(os.path.join('aichal', 'val', imname))
    im = np.array(im, dtype=np.float32)
    im = im[:,:,::-1]
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = im.transpose((2,0,1))

    f = np.zeros((52, 52))
    cv2.circle(f, (25, 25), 25, 1, -1)

    kps = list()

    for box in boxes:
        if box[-1] >= box_thresh:

            imCrop = im[:,box[1]:box[3], box[0]:box[2]]

            # shape for input (data blob is N x C x H x W), set data
            net.blobs['data'].reshape(1, *imCrop.shape)
            net.blobs['data'].data[...] = imCrop

            t['forward'].tic()
            net.forward()
            t['forward'].toc()

            t['process'].tic()
            score = net.blobs['prob'].data.copy() # 14 x 2 x W x H
            kp = []
            prob = 0.0
            for i in range(14):
                score_i = score[i] # 2 x W x H
                score_i = score_i.transpose((1, 2, 0)) # W x H x 2
                score_i = score_i[:,:,1]

                t['conv'].tic()
                score_i = signal.fftconvolve(score_i, f, 
                        mode='same') / (25 * 25 * 3.14)
                t['conv'].toc()

                p  = np.max(score_i)
                x, y = np.unravel_index(np.argmax(score_i), score_i.shape)
                kp.append([y, x])
                prob += p / 14


            kp_ann = {}
            kp_ann['kp'] = kp
            kp_ann['prob'] = prob
            kp_ann['box'] = box
            kps.append(kp_ann)
            t['process'].toc()

    image['kps'] = kps
    print '{}  forward: {:.3f}s  process: {:.3f}.s  conv: {:.3f}.s'.format(
        idx, t['forward'].average_time, 
        t['process'].average_time, t['conv'].average_time)

    return image



def main():
    caffe.set_mode_gpu()
    net = caffe.Net('deploy.prototxt', './kp_snapshot/v2_iter_100000.caffemodel', caffe.TEST)

    val = np.loadtxt('./aichal/val.txt', dtype=str)
    pkl = 'aichalClusterBoxVal.pkl'
    with open(pkl, 'rb') as fid:
        boxes = cPickle.load(fid)[1]

    data = []
    box_thresh = 0.5

    with open ('pred_cluster_fcn_' + str(box_thresh) +'.pkl', 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)


    t = {'forward':Timer(), 'process':Timer(), 'conv':Timer()}

    for i, line in enumerate(val):
        data.append(getKeypoint(line, boxes[i], box_thresh, net, t, i))


    with open ('pred_cluster_fcn_' + str(box_thresh) +'.pkl', 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
    
