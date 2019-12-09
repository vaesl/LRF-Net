from __future__ import print_function
import pickle
import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from torch.autograd import Variable
from collections import OrderedDict
from data import COCOdir
from data import COCODetection, BaseTransform, COCO_300, COCO_512

from layers.functions import Detect, PriorBox
from utils.nms_wrapper import nms


parser = argparse.ArgumentParser(description='Testing Learning Rich Features Network')
parser.add_argument('-s', '--size', default='300', help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='COCO', help='Currently we only provide results on MS COCO')
parser.add_argument('-m', '--trained_model', default='weights/COCO/LRF_COCO_300/Final_LRF_vgg_COCO.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to test model')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

cfg = COCO_300 if args.size == '300' else COCO_512

if args.dataset == 'COCO':
    if args.size == '300':
        from models.LRF_COCO_300 import build_net
    else:
        from models.LRF_COCO_512 import build_net
else:
    print('Unkown Dataset!')

priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward(), volatile=True)
priors = priors.cpu() if not args.cuda else priors


def test_net(save_folder, net, detector, cuda, testset, transform, top_k=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    num_images = len(testset)
    num_classes = 81
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]

    det_file = os.path.join(save_folder, 'detections.pkl')

    for i in range(num_images):
        img = testset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0), volatile=True)
        x = x.cuda() if cuda else x

        out = net(x)
        boxes, scores = detector.forward(out, priors)

        boxes = boxes[0]
        scores = scores[0]
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale   back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)

            cpu = False
            keep = nms(c_dets, 0.45, force_cpu=cpu)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets

        if top_k > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > top_k:
                image_thresh = np.sort(image_scores)[-top_k]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, protocol=2)
    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    # load net
    img_dim = (300, 512)[args.size == '512']
    num_classes = (21, 81)[args.dataset == 'COCO']
    net = build_net('test', img_dim, num_classes)    # initialize detector

    state_dict = torch.load(args.trained_model)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if args.dataset == 'COCO' else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading LRFNet model!')
    print(net)

    if args.dataset == 'COCO':
        testset = COCODetection(
            COCOdir, [('2014', 'minival')], None)
            # COCOdir, [('2015', 'test-dev')], None)
    else:
        print('Only COCO dataset is supported now!')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    top_k = 200
    detector = Detect(num_classes, 0, cfg)
    save_folder = os.path.join(args.save_folder, args.dataset)
    rgb_means = (104, 117, 123)
    test_net(save_folder, net, detector, args.cuda, testset, BaseTransform(net.size, rgb_means, (2, 0, 1)),
             top_k, thresh=0.01)
