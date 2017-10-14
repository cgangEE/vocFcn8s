#!/usr/bin/env python
import cPickle
import matplotlib.pyplot as plt
import json

def showProbHist(data):
    prob = []
    for line in data:
        for kp in line['kps']:
            prob.append(kp['prob'])
    plt.hist(prob, bins=100)
    plt.show()


def main():


    with open('pred_cluster_fcn_0.5.pkl', 'rb') as f:
        data = cPickle.load(f)
#    showProbHist(data)

    out = []
    thresh = 0.05

    for line in data:
        image = {}
        image['image_id'] = line['image_id']    

        kps = line['kps']
        pred_kps = dict()

        for i, kp_ann in enumerate(kps):
            if kp_ann['prob'] < thresh:
                continue
            kp = kp_ann['kp']
            box = kp_ann['box']

            pred_kp = []
            for p in kp:
                pred_kp += [p[0] + box[0]]
                pred_kp += [p[1] + box[1]]
                pred_kp += [1]

            pred_kps['human' + str(i+1)] = map(int, pred_kp)

        image['keypoint_annotation'] = pred_kps
        out.append(image)


    

    with open('pred_fcn_0.5_' + str(thresh) +'.json', 'w') as f:
        json.dump(out, f)

if __name__ == '__main__':
    main()

