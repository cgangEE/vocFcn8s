#!/usr/bin/env python
import numpy as np
import cv2

def gao():
    x = np.zeros((100, 100, 1))
    print(np.sum(x))
    cv2.circle(x, (50, 50), 20, 255, -1)
    print(x[50])
    print(np.sum(x))
    cv2.imshow('x', x)
    cv2.waitKey()

def gao2():
    x = np.zeros((3, 4, 5))
    x[1:2,:,:] = 2
    x[:,2:3,:] = 4
    x[:,:,4:5] = 1
    print(x[2])
    x = x.transpose(1, 2, 0)
    print(x[:,:,2])

gao()
