# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os
import random
import warnings

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import tensorflow as tf
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from bases.trainer_base import TrainerBase
from root_dir import ROOT_DIR
from utils.np_utils import prp_2_oh_array
from utils.utils import mkdir_if_not_exist



class TripletTrainer(TrainerBase):
    def __init__(self, model, data, config):
        super(TripletTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        train_dir = os.path.join(ROOT_DIR, self.config.tb_dir, "train")
        mkdir_if_not_exist(train_dir)
        self.callbacks.append(
            TensorBoard(
                log_dir=train_dir,
                write_images=True,
                write_graph=True,
            )
        )

        #self.callbacks.append(FPRMetric())
        #self.callbacks.append(FPRMetricDetail())

    def train(self):
        x_train = self.data[0][0]
        y_train = self.data[0][1]
        #y_train = np.argmax(self.data[0][1], axis=1)
        x_test = self.data[1][0]
        y_test = self.data[1][1]
        #y_test = np.argmax(self.data[1][1], axis=1)

        clz_size = len(np.unique(y_train))
        print "[INFO] trainer - 类别数: %s" % clz_size
        print(y_train)
        y_train = np.array(list(y_train))
        print(y_train)

        tr_pairs = self.create_pairs(x_train, y_train)
        te_pairs = self.create_pairs(x_test, y_test)

        #print(tr_pairs)
        anc_ins = tr_pairs[:, 0]
        pos_ins = tr_pairs[:, 1]
        neg_ins = tr_pairs[:, 2]

        anc_ins_list = []
        pos_ins_list = []
        neg_ins_list = []

        for i in range(len(anc_ins)):
            #print(list(anc_ins[i]))
            anc_ins_list.append(list(anc_ins[i]))
        
        for i in range(len(pos_ins)):
            #print(list(anc_ins[i]))
            pos_ins_list.append(list(pos_ins[i]))
        
        for i in range(len(neg_ins)):
            #print(list(anc_ins[i]))
            neg_ins_list.append(list(neg_ins[i]))
        
        anc_ins = np.array(anc_ins_list)
        pos_ins = np.array(pos_ins_list)
        neg_ins = np.array(neg_ins_list)
        print(anc_ins.shape)
        print(len(anc_ins[0]))



        X = {
            'anc_input': anc_ins,
            'pos_input': pos_ins,
            'neg_input': neg_ins
        }

        anc_ins_te = te_pairs[:, 0]
        pos_ins_te = te_pairs[:, 1]
        neg_ins_te = te_pairs[:, 2]


        anc_ins_te_list = []
        pos_ins_te_list = []
        neg_ins_te_list = []

        for i in range(len(anc_ins_te)):
            #print(list(anc_ins[i]))
            anc_ins_te_list.append(list(anc_ins_te[i]))
        
        for i in range(len(pos_ins_te)):
            #print(list(anc_ins[i]))
            pos_ins_te_list.append(list(pos_ins_te[i]))
        
        for i in range(len(neg_ins_te)):
            #print(list(anc_ins[i]))
            neg_ins_te_list.append(list(neg_ins_te[i]))
        
        anc_ins_te = np.array(anc_ins_te_list)
        pos_ins_te = np.array(pos_ins_te_list)
        neg_ins_te = np.array(neg_ins_te_list)

        X_te = {
            'anc_input': anc_ins_te,
            'pos_input': pos_ins_te,
            'neg_input': neg_ins_te
        }

        print(y_train.shape)

        self.model.fit(
            X, 
            y_train,
            batch_size=32,
            epochs=14,
            validation_data=[X_te,y_test],
            verbose=1,
            callbacks=self.callbacks)

        self.model.save(os.path.join(self.config.cp_dir, "triplet_loss_model.h5"))  # 存储模型

        y_pred = self.model.predict(X_te)  #验证模型
        
        self.show_acc_facets(y_pred, y_pred.shape[0] / clz_size, clz_size)

        y_pred_label=y_pred[:,384:]
        print(y_pred_label.shape)
        y_pred_label_list = []
        for i in y_pred_label:
            #print(i)
   
            if i[0]>i[1]:
                pred_label = 0
            else:
                pred_label = 1
            y_pred_label_list.append(pred_label)

        y_test_list=[]

        for j in y_test:
            #print('j',j)
            #print(j[0])
            if j[0]>j[1]:
                test_label = 0
            else:
                test_label = 1

            y_test_list.append(test_label)
   
        accuracy = accuracy_score(y_test_list, y_pred_label_list)

        print("Accuracy:", accuracy)
        f_score = f1_score(y_test_list, y_pred_label_list)
        precision = precision_score(y_test_list, y_pred_label_list)
        recall = recall_score(y_test_list, y_pred_label_list)
        print("f_score,precision, recall:",f_score,precision, recall)


        #for p, r, f, s in zip(precision, recall, f_score, support):
        #print " — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f - ins %s" % 
        

    @staticmethod
    def show_acc_facets(y_pred, n, clz_size):
        """
        展示模型的准确率
        :param y_pred: 测试结果数据组
        :param n: 数据长度
        :param clz_size: 类别数
        :return: 打印数据
        """
        print "[INFO] trainer - n_clz: %s" % n
        for i in range(clz_size):
            print "[INFO] trainer - clz %s" % i
            final = y_pred[n * i:n * (i + 1), :]
            anchor, positive, negative = final[:, 0:128], final[:, 128:256], final[:, 256:384]

            pos_dist = np.sum(np.square(anchor - positive), axis=-1, keepdims=True)
            neg_dist = np.sum(np.square(anchor - negative), axis=-1, keepdims=True)
            basic_loss = pos_dist - neg_dist
            r_count = basic_loss[np.where(basic_loss < 0)].shape[0]
            print "[INFO] trainer - distance - min: %s, max: %s, avg: %s" % (
                np.min(basic_loss), np.max(basic_loss), np.average(basic_loss))
            print "[INFO] acc: %s" % (float(r_count) / float(n))
            print ""
   

    @staticmethod
    def create_pairs(x,y):
        pairs = []
        print(len(x))
        for i in range(len(x)):
            #print("y[i]",y[i])
            y_temp = list(y[i])
            if y_temp[0]<y_temp[1]:
                pairs += [[x[i][0:2048],x[i][3072:4096],x[i][2048:3072]]]   
            else:
                pairs += [[x[i][0:2048],x[i][2048:3072],x[i][3072:4096]]]            
            
        return np.array(pairs)


class FPRMetric(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            val_y, prd_y, average='macro')
        print " — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f" % (f_score, precision, recall)


class FPRMetricDetail(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, support = precision_recall_fscore_support(val_y, prd_y)

        for p, r, f, s in zip(precision, recall, f_score, support):
            print " — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f - ins %s" % (f, p, r, s)
