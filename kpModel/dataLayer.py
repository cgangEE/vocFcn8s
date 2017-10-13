import caffe
import numpy as np
from PIL import Image
import random, json, os, cv2


class TrainDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.dir = params['dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', False)
        self.seed = params.get('seed', None)


        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.dir, self.split)
        self.indices = open(split_f, 'r').read().splitlines()

        split_f_ann  = '{}/{}.json'.format(self.dir, self.split)
        with open(split_f_ann) as f:
            self.ann = json.load(f)


        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        self.idx = 0
        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)



    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image()
        self.label = self.load_label(self.indices[self.idx], self.data)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(*self.label.shape)

    def next(self):

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        self.next()


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self):
        while True:
            try:
                idx = self.indices[self.idx]
                im = Image.open(idx)
                break
            except IOError:
                self.next()

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx, im):
        ret = np.zeros([14, im.shape[1], im.shape[2], 1])
        kp = self.ann[idx]


        for i in range(14):
            x, y, z = kp[i * 3: (i + 1) * 3]
            if z <= 2:
                cv2.circle(ret[i], (x, y), 25, 1, -1)
            elif z == 3:
                ret[i] = np.ones((im.shape[1], im.shape[2], 1)) * 255.0

        ret = ret.transpose(0, 3, 1, 2)
        return ret

