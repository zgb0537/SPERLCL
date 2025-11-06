# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from keras.datasets import mnist
from keras.utils import to_categorical

from bases.data_loader_base import DataLoaderBase
import numpy as np
import os
import random
random.seed(100)

class TripletDL(DataLoaderBase):
    def __init__(self, config=None):
        super(TripletDL, self).__init__(config)
        #(self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        print(os.getcwd())
        self.X_train = np.load("./data_loaders/train_data.npy")
        self.y_train = np.load("./data_loaders/train_label.npy")
        self.X_test = np.load("./data_loaders/test_data.npy")
        self.y_test = np.load("./data_loaders/test_label.npy")
        # 展平数据，for RNN，感知机
        #self.X_train = self.X_train.reshape((-1, 28 * 28))
        #self.X_test = self.X_test.reshape((-1, 28 * 28))

        # 图片数据，for CNN
        #self.X_train = self.X_train.reshape(self.X_train.shape[0], 32, 32, 1)
        #self.X_test = self.X_test.reshape(self.X_test.shape[0], 32, 32, 1)
        print(self.y_train[0])
        self.y_train = to_categorical(self.y_train)  #to_categorical 实现独热编码
        print("y_train", self.y_train[0]) 
        self.y_test = to_categorical(self.y_test)

        print "[INFO] X_train.shape: %s, y_train.shape: %s" \
              % (str(self.X_train.shape), str(self.y_train.shape))
        print "[INFO] X_test.shape: %s, y_test.shape: %s" \
              % (str(self.X_test.shape), str(self.y_test.shape))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
