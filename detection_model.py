# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:32:33 2018

@author: Weiyu_Lee
"""

import os
import sys
sys.path.append('./utility')

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import scipy.misc
import pickle
import random

import DE_model_zoo
import utils

from tensorflow.examples.tutorials.mnist import input_data

class DETECTION_MODEL(object):
    def __init__(self, 
                 sess, 
                 mode=None,
                 is_train=True,
                 iteration=400000,
                 curr_iteration=0,                 
                 batch_size=32,
                 image_size=256,
                 project_image_size=384,
                 code_size=[64,64,4],                 
                 learning_rate=1e-4,
                 checkpoint_dir=None, 
                 ckpt_name=None,
                 log_dir=None,
                 output_dir=None,
                 model_ticket=None,
                 datasetroot=None):                 
        """
        Initial function
          
        Args:
            image_size: training or testing input image size. 
                        (if scale=3, image size is [33x33].)
            label_size: label image size. 
                        (if scale=3, image size is [21x21].)
            batch_size: batch size
            color_dim: color dimension number. (only Y channel, color_dim=1)
            checkpoint_dir: checkpoint directory
            output_dir: output directory
        """  
        
        self.sess = sess
        
        self.mode = mode

        self.is_train = is_train      

        self.iteration = iteration
        self.curr_iteration = curr_iteration
        
        self.batch_size = batch_size
        self.image_size = image_size
        self.code_size = code_size
        self.project_image_size = project_image_size

        self.learning_rate = learning_rate 
    
        self.checkpoint_dir = checkpoint_dir
        self.ckpt_name = ckpt_name
        
        self.log_dir = log_dir
        self.output_dir = output_dir
               
        self.datasetroot = datasetroot
        
        self.model_ticket = model_ticket
        
        self.model_list = ["baseline", "alex_net"]
               
        self.build_model()        
    
    def build_model(self):###              
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            fn = getattr(self, "build_" + self.model_ticket)
            model = fn()
            return model    
        
    def train(self):
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            fn = getattr(self, "train_" + self.model_ticket)
            function = fn()
            return function                 

    def load_pickle_data(self):
        
        f = open(os.path.join(self.datasetroot, "normal_data.pkl"), "rb")
        n_image = np.array(pickle.load(f))
        n_code = np.array(pickle.load(f))
        f.close()
                
        f = open(os.path.join(self.datasetroot, "abnormal_data.pkl"), "rb")
        ab_image = np.array(pickle.load(f))
        ab_code = np.array(pickle.load(f))
        f.close()        
        
        return n_image, n_code, ab_image, ab_code

    def load_ckpt(self, checkpoint_dir, ckpt_name=""):
        """
        Load the checkpoint. 
        According to the scale, read different folder to load the models.
        """     
        
        print(" [*] Reading checkpoints...")
        if ckpt_name == "":
            model_dir = "%s_%s_%s" % ("srcnn", "scale", self.scale)
        else:
            model_dir = ckpt_name
            
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        print("Current checkpoints: [{}]".format(os.path.join(checkpoint_dir, ckpt_name)))
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        
            return True
        
        else:
            return False          

    def get_batch_data_idx(self, num, batch_size):
        
        idx = list(range(0, num))
        random.shuffle(idx)
        batch_data_idx = idx[:batch_size]
        
        return batch_data_idx
                
    def save_ckpt(self, checkpoint_dir, ckpt_name, step):
        """
        Save the checkpoint. 
        According to the scale, use different folder to save the models.
        """          
        
        print(" [*] Saving checkpoints...step: [{}]".format(step))
        model_name = ckpt_name
        
        if ckpt_name == "":
            model_dir = "%s_%s_%s" % ("srcnn", "scale", self.scale)
        else:
            model_dir = ckpt_name
        
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def save_best_ckpt(self, checkpoint_dir, ckpt_name, loss, step):
        """
        Save the checkpoint. 
        According to the scale, use different folder to save the models.
        """          
        
        print(" [*] Saving best checkpoints...step: [{}]\n".format(step))
        model_name = ckpt_name + "_{}".format(loss)
        
        if ckpt_name == "":
            model_dir = "%s_%s_%s" % ("srcnn", "scale", self.scale)
        else:
            model_dir = ckpt_name
        
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        checkpoint_dir = os.path.join(checkpoint_dir, "best_performance")
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.best_saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)                

    def save_output(self, encode_img, decode_img, ckpt_name, iteration):
        
        if not os.path.exists('./output_{}'.format(ckpt_name)):
            os.makedirs('./output_{}'.format(ckpt_name))                
            
        scipy.misc.imsave('./output_{}/encode_input_{}.png'.format(ckpt_name, iteration), encode_img.squeeze())
        scipy.misc.imsave('./output_{}/model_output_{}.png'.format(ckpt_name, iteration), decode_img.squeeze())                 
    
    def build_baseline(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='input')
        self.label = tf.placeholder(tf.float32, [None, self.code_size[0]*self.code_size[1]*self.code_size[2]], name='label')
        self.code_image = tf.reshape(self.label, [tf.shape(self.label)[0], self.code_size[0], self.code_size[1], self.code_size[2]])
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = DE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        self.pred = mz.build_model()       

        self.pred_reshape = tf.reshape(self.pred, [tf.shape(self.input)[0], self.code_size[0], self.code_size[1], self.code_size[2]])
        
        self.l2_loss = tf.pow(self.pred - self.label, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.loss, collections=['train'])
            tf.summary.image("input_image", self.input, collections=['train'])
            tf.summary.image("code_image", self.code_image, collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred_reshape, 0, 500), collections=['train'])
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.loss, collections=['test'])
            tf.summary.image("input_image", self.input, collections=['test'])
            tf.summary.image("code_image", self.code_image, collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred_reshape, 0, 500), collections=['test'])
            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline(self):
        """
        Training process.
        """     
        print("Loading data...")
        n_image, n_code, ab_image, ab_code = self.load_pickle_data()
        
        print("Spliting training & testing data...")
        test_num = round(len(ab_image)*0.2)
        train_num = len(ab_image) - test_num
        
        test_ab_idx = list(range(0, len(ab_image)))       
        random.shuffle(test_ab_idx)
        test_ab_image = ab_image[test_ab_idx[:test_num]]
        test_ab_code = ab_code[test_ab_idx[:test_num]]
        train_ab_image = ab_image[test_ab_idx[test_num:]]
        train_ab_code = ab_code[test_ab_idx[test_num:]]

        test_n_idx = list(range(0, len(n_image)))
        random.shuffle(test_n_idx)
        test_n_image = n_image[test_n_idx[:test_num]]
        test_n_code = n_code[test_n_idx[:test_num]]
        train_n_image = n_image[test_n_idx[test_num:test_num+train_num]]
        train_n_code = n_code[test_n_idx[test_num:test_num+train_num]]

        print("Test normal:abnormal sample: [{}:{}]".format(len(test_n_image), len(test_ab_image)))
        print("Train normal:abnormal sample: [{}:{}]".format(len(train_n_image), len(train_ab_image))) 
        
        print("Training...")
        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
       
        #util = utils.utility(self.image_size, self.image_size, self.project_image_size)
        
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
          
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 1000      
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            n_idx = self.get_batch_data_idx(len(train_n_image), self.batch_size//2)
            ab_idx = self.get_batch_data_idx(len(train_ab_image), self.batch_size//2)

            batch_train_data = np.concatenate((train_ab_image[ab_idx], train_n_image[n_idx]))
            batch_train_label_data = np.concatenate((train_ab_code[ab_idx], train_n_code[n_idx]))
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_train_data,
                                           self.label: batch_train_label_data,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss],
                                                        feed_dict={
                                                                    self.input: batch_train_data,
                                                                    self.label: batch_train_label_data,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                                                                                                                   

                n_idx = self.get_batch_data_idx(len(test_n_image), self.batch_size//2)
                ab_idx = self.get_batch_data_idx(len(test_ab_image), self.batch_size//2)
    
                batch_test_data = np.concatenate((test_ab_image[ab_idx], test_n_image[n_idx]))
                batch_test_label_data = np.concatenate((test_ab_code[ab_idx], test_n_code[n_idx]))
                
                test_sum, test_loss = \
                                                        self.sess.run([self.merged_summary_test, 
                                                        self.loss],
                                                        feed_dict={
                                                                    self.input: batch_test_data,
                                                                    self.label: batch_test_label_data,
                                                                  })                    
                
#                if it % (save_it*10) == 0 and it != 0:
#                    decode_output = util.projection_reverse(test_pred, 1)                   
#                    self.save_output(batch_train_data[0], decode_output[0], self.ckpt_name, it)
                                                                     
                print("Iter: [{}], Train: [{}], Test: [{}]".format((it+1), train_loss, test_loss))       
                
                summary_writer.add_summary(test_sum, it)        
                summary_writer.add_summary(train_sum, it)              
    def build_alex_net(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='input')
        self.label = tf.placeholder(tf.float32, [None, self.code_size[0]*self.code_size[1]*self.code_size[2]], name='label')
        self.code_image = tf.reshape(self.label, [tf.shape(self.label)[0], self.code_size[0], self.code_size[1], self.code_size[2]])
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = DE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        self.pred = mz.build_model()       

        self.pred_reshape = tf.reshape(self.pred, [tf.shape(self.input)[0], self.code_size[0], self.code_size[1], self.code_size[2]])
        
        self.l2_loss = tf.pow(self.pred - self.label, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.loss, collections=['train'])
            tf.summary.image("input_image", self.input, collections=['train'])
            tf.summary.image("code_image", self.code_image, collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred_reshape, 0, 500), collections=['train'])
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.loss, collections=['test'])
            tf.summary.image("input_image", self.input, collections=['test'])
            tf.summary.image("code_image", self.code_image, collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred_reshape, 0, 500), collections=['test'])
            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_alex_net(self):
        """
        Training process.
        """     
        print("Loading data...")
        n_image, n_code, ab_image, ab_code = self.load_pickle_data()
        
        print("Spliting training & testing data...")
        test_num = round(len(ab_image)*0.2)
        train_num = len(ab_image) - test_num
        
        test_ab_idx = list(range(0, len(ab_image)))       
        random.shuffle(test_ab_idx)
        test_ab_image = ab_image[test_ab_idx[:test_num]]
        test_ab_code = ab_code[test_ab_idx[:test_num]]
        train_ab_image = ab_image[test_ab_idx[test_num:]]
        train_ab_code = ab_code[test_ab_idx[test_num:]]

        test_num = round(len(n_image)*0.2)
        train_num = len(n_image) - test_num
        
        test_n_idx = list(range(0, len(n_image)))
        random.shuffle(test_n_idx)
        test_n_image = n_image[test_n_idx[:test_num]]
        test_n_code = n_code[test_n_idx[:test_num]]
        train_n_image = n_image[test_n_idx[test_num:test_num+train_num]]
        train_n_code = n_code[test_n_idx[test_num:test_num+train_num]]

        print("Test normal:abnormal sample: [{}:{}]".format(len(test_n_image), len(test_ab_image)))
        print("Train normal:abnormal sample: [{}:{}]".format(len(train_n_image), len(train_ab_image))) 
        
        print("Training...")
        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
       
        #util = utils.utility(self.image_size, self.image_size, self.project_image_size)
        
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
          
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 1000      
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            n_idx = self.get_batch_data_idx(len(train_n_image), self.batch_size//4)
            ab_idx = self.get_batch_data_idx(len(train_ab_image), self.batch_size//4*3)

            batch_train_data = np.concatenate((train_ab_image[ab_idx], train_n_image[n_idx]))
            batch_train_label_data = np.concatenate((train_ab_code[ab_idx], train_n_code[n_idx]))
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_train_data,
                                           self.label: batch_train_label_data,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss],
                                                        feed_dict={
                                                                    self.input: batch_train_data,
                                                                    self.label: batch_train_label_data,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                                                                                                                   

                n_idx = self.get_batch_data_idx(len(test_n_image), self.batch_size//4)
                ab_idx = self.get_batch_data_idx(len(test_ab_image), self.batch_size//4*3)
    
                batch_test_data = np.concatenate((test_ab_image[ab_idx], test_n_image[n_idx]))
                batch_test_label_data = np.concatenate((test_ab_code[ab_idx], test_n_code[n_idx]))
                
                test_sum, test_loss, test_pred = \
                                                        self.sess.run([self.merged_summary_test, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_test_data,
                                                                    self.label: batch_test_label_data,
                                                                  })                    
                
#                if it % (save_it*10) == 0 and it != 0:
#                    decode_output = util.projection_reverse(test_pred, 1)                   
#                    self.save_output(batch_train_data[0], decode_output[0], self.ckpt_name, it)
                                                                     
                print("Iter: [{}], Train: [{}], Test: [{}]".format((it+1), train_loss, test_loss))       
                
                summary_writer.add_summary(test_sum, it)        
                summary_writer.add_summary(train_sum, it)                        