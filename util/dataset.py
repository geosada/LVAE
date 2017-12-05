import tensorflow as tf
import numpy as np
import sys, os, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../util')


class Dataset(object):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size 

        if self.dataset == 'MNIST':
            n_train, n_test = 55000, 10000
            _h, _w, _c = 28,28,1
            _img_size = _h*_w*_c
            _l = 10
            _is_3d = True
        else:
            sys.exit('[ERROR] not implemented yet.')

        self.h = _h
        self.w = _w
        self.c = _c
        self.l = _l
        self.is_3d     = _is_3d 
        self.img_size  = _img_size
        self.n_train   = n_train
        self.n_test    = n_test
        self.n_batches_train = int(n_train/batch_size)
        self.n_batches_test  = int(n_test/batch_size)

    def get_tfrecords(self):

        # xtrain: all records
        # *_l   : partial records
        from mnist import inputs
        xtrain,_           = inputs(self.batch_size, 'train')
        xtrain_l, ytrain_l = inputs(self.batch_size, 'train_labeled')
        xtest , ytest      = inputs(self.batch_size, 'test')

        return (xtrain_l, ytrain_l), xtrain, (xtest , ytest)
