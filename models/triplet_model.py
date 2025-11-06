# -- coding: utf-8 --

import os, keras
from keras import Input, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, K, merge, Concatenate, Multiply, Lambda
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np
from bases.model_base import ModelBase
import tensorflow as tf
from keras import backend as kk


class TripletModel(ModelBase):
    """
    TripletLoss模型
    """

    MARGIN = 1.0  # 超参

    def __init__(self, config):
        super(TripletModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = self.triplet_loss_model()  # 使用Triplet Loss训练Model

    def triplet_loss_model(self):
          
        anc_input = Input(shape=(2048,), name='anc_input')  # anchor
        pos_input = Input(shape=(1024,), name='pos_input')  # positive
        neg_input = Input(shape=(1024,), name='neg_input')  # negative

        shared_model = self.base_model()  # 共享模型
        shared_model1 = self.base_model1()
        
        std_out = shared_model(anc_input)
        pos_out = shared_model1(pos_input)
        neg_out = shared_model1(neg_input)

        print "[INFO] model - 锚shape: %s" % str(std_out.get_shape())
        print "[INFO] model - 正shape: %s" % str(pos_out.get_shape())
        print "[INFO] model - 负shape: %s" % str(neg_out.get_shape())
        out = Dense(2, activation='softmax')(std_out)

        output = Concatenate()([std_out, pos_out, neg_out, out])  # 连接
        model = Model(inputs=[anc_input, pos_input, neg_input], outputs=output)

        plot_model(model, to_file=os.path.join(self.config.img_dir, "triplet_loss_model.png"),
                   show_shapes=True)  # 绘制模型图
        model.compile(loss=self.triplet_loss, optimizer=Adam())
        return model

    @staticmethod
    def triplet_loss(y_true, y_pred):
        """
        Triplet Loss的损失函数
        """
        #print("y_true shape",y_true.shape)
        anc, pos, neg, pre_label = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:384] , y_pred[:,384:]
        anc = Dense(128, activation='tanh')(anc)
        pos = Dense(128, activation='tanh')(pos)
        neg = Dense(128, activation='tanh')(neg)
        # 欧式距离
        pos_dist = K.sum(K.square(anc - pos), axis=-1, keepdims=True)
        neg_dist = K.sum(K.square(anc - neg), axis=-1, keepdims=True)
        basic_loss = pos_dist - neg_dist + TripletModel.MARGIN

        loss0 = K.sum(K.maximum(basic_loss, 0.0))
        print(pre_label.shape)
        
        print(y_true.shape)

        loss1=tf.keras.losses.categorical_crossentropy(y_true, pre_label)
        #loss1 = K.sqrt(K.mean(K.square(pre_label - y_true), axis=-1))
        #loss1 = -y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)
        #loss1=k.mean(loss1)

        c= 20
        loss_all = 0.01*loss0+loss1

        print "[INFO] model - triplet_loss shape: %s" % str(loss_all.shape)
        return loss_all

    def base_model(self):
        """
        Triplet Loss的基础网络，可以替换其他网络结构
        """
       
        ins_input = Input(shape=(2048,))
        x1 = Lambda(lambda x: x[:, :1024])(ins_input)
        x2 = Lambda(lambda x: x[:, 1024:])(ins_input)
        x = Multiply()([x1,x2])
        x = Dropout(0.1)(x)
        out = Dense(128, activation='relu')(x)

        model = Model(ins_input, out)
        plot_model(model, to_file=os.path.join(self.config.img_dir, "base_model.png"), show_shapes=True)  # 绘制模型图
        return model

    def base_model1(self):
        """
        Triplet Loss的基础网络，可以替换其他网络结构
        
        ins_input = Input(shape=(28, 28, 1))
        x = Conv2D(32, kernel_size=(3, 3), kernel_initializer='random_uniform', activation='relu')(ins_input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, kernel_size=(3, 3), kernel_initializer='random_uniform', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        out = Dense(128, activation='relu')(x)

        model = Model(ins_input, out)
        plot_model(model, to_file=os.path.join(self.config.img_dir, "base_model.png"), show_shapes=True)  # 绘制模型图
        return model
        """
        ins_input = Input(shape=(1024,))
        x = Dropout(0.1)(ins_input)
        out = Dense(128, activation='relu')(x)

        model = Model(ins_input, out)
        plot_model(model, to_file=os.path.join(self.config.img_dir, "base_model1.png"), show_shapes=True)  # 绘制模型图
        return model