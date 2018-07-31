# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:20:02 2017

@author: Weiyu_Lee
"""

import os
import sys
sys.path.append('./utility')

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import scipy.misc
import math
import random 

import AE_model_zoo

import utils

#from utils import (
#    get_batch_data,
#    log10
#)

#from tensorflow.examples.tutorials.mnist import input_data

class AUTOENCODER_MODEL(object):
    def __init__(self, 
                 sess, 
                 mode=None,
                 is_train=True,
                 iteration=100000,
                 curr_iteration=0,                 
                 batch_size=128,
                 image_size=32,
                 project_image_size=math.sqrt((32**2)*2),
                 label_size=20, 
                 learning_rate=1e-4,
                 checkpoint_dir=None, 
                 ckpt_name=None,
                 log_dir=None,
                 output_dir=None,
                 train_dir=None,
                 test_dir=None,
                 model_ticket=None,
                 test_input_path=None,
                 test_label_path=None):                 
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
        self.project_image_size = project_image_size

        self.learning_rate = learning_rate 
    
        self.checkpoint_dir = checkpoint_dir
        self.ckpt_name = ckpt_name
        
        self.log_dir = log_dir
        self.output_dir = output_dir
        
        self.train_dir = train_dir
        self.test_dir = test_dir       
               
        self.model_ticket = model_ticket

        self.test_input_path = test_input_path
        self.test_label_path = test_label_path
        
        self.model_list = ["baseline", "baseline_v2", "baseline_v3", "baseline_v4", "baseline_v5", "baseline_v5_flatten", "baseline_v6_flatten", 
                           "baseline_end2end", "baseline_end2end_2D", "baseline_end2end_2D_v2"]
               
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

    def load_image_data(self):
        
        input_data = []
        label_data = []
        image_names = []
        
        input_data_list = os.listdir(self.test_input_path)
        
        for f in input_data_list:
            curr_input = scipy.misc.imread(os.path.join(self.test_input_path, f))
            curr_label = scipy.misc.imread(os.path.join(self.test_label_path, f))

            file_name = os.path.splitext(f)[0]

            input_data.append(curr_input)
            label_data.append(curr_label)            
            image_names.append(file_name)
                    
        return input_data, label_data, image_names

    def split_image(self, input, label, image_name, sub_size):
        
        height, width = label.shape
        
        output_image = []
        output_label = []
        output_check_list = np.zeros([(height//sub_size)*(width//sub_size) + height//sub_size + width//sub_size + 1])
        output_meta_data = []
        
        patch_count = 0
        residual_h_count = 0
        residual_w_count = 0
        
        for h in range(0, height-sub_size, sub_size):
            for w in range(0, width-sub_size, sub_size):    
                output_image.append(input[h:h+sub_size, w:w+sub_size])
                output_label.append(label[h:h+sub_size, w:w+sub_size])
                
                if output_label[-1].sum() > 50: 
                    output_check_list[patch_count] = 1                        
                    output_meta_data.append((image_name, [h,w], "regular_grid", "abnormal"))        
                else:
                    output_meta_data.append((image_name, [h,w], "regular_grid", "normal"))                    
                    
                patch_count += 1
        
        # Add the residual parts
        if (height % sub_size != 0):
           h = height-sub_size
           for  w in range(0, width-sub_size, sub_size):    
               output_image.append(input[h:h+sub_size, w:w+sub_size])
               output_label.append(label[h:h+sub_size, w:w+sub_size])
               residual_h_count += 1

               if output_label[-1].sum() > 50: 
                   output_check_list[patch_count] = 1                        
                   output_meta_data.append((image_name, [h,w], "residual_grid", "abnormal"))        
               else:
                   output_meta_data.append((image_name, [h,w], "residual_grid", "normal"))     
                    
               patch_count += 1               
               
        if (width % sub_size != 0):
           w = width-sub_size
           for  h in range(0, height-sub_size, sub_size):
               output_image.append(input[h:h+sub_size, w:w+sub_size])
               output_label.append(label[h:h+sub_size, w:w+sub_size]) 
               residual_w_count += 1

               if output_label[-1].sum() > 50: 
                   output_check_list[patch_count] = 1                        
                   output_meta_data.append((image_name, [h,w], "residual_grid", "abnormal"))        
               else:
                   output_meta_data.append((image_name, [h,w], "residual_grid", "normal"))     
                    
               patch_count += 1                   

        # Add the right-bottom residual part
        output_image.append(input[height-sub_size:height+sub_size, width-sub_size:width+sub_size])
        output_label.append(label[height-sub_size:height+sub_size, width-sub_size:width+sub_size])            
        
        if output_label[-1].sum() > 50: 
            output_check_list[patch_count] = 1                        
            output_meta_data.append((image_name, [height-sub_size, width-sub_size], "residual_grid_rb", "abnormal"))        
        else:
            output_meta_data.append((image_name, [height-sub_size, width-sub_size], "residual_grid_rb", "normal"))                   
        
        patch_count += 1            
        
        print("output_label shape: {}, residual parts: [h,w] = [{}, {}]".format(np.array(output_label).shape, residual_h_count, residual_w_count))
        print("output_meta_data shape: {}".format(len(output_meta_data)))
        
        return output_image, output_label, output_check_list, output_meta_data

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
        scipy.misc.imsave('./output_{}/decode_output_{}.png'.format(ckpt_name, iteration), decode_img.squeeze())                 

    def build_baseline(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], name='images')
        
#        bsize, a, b, c = self.input.get_shape().as_list()
#        if bsize == None:
#            bsize = -1
#        self.fc_input = tf.reshape(self.input, [bsize, a*b*c])    
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        self.code, self.pred = mz.build_model()       
        
        #self.pred_image = tf.reshape(self.pred, tf.shape(self.input))

        #self.code_image = tf.reshape(self.code, [-1, 32, 32, 1])
        
        self.mask = tf.cast(tf.greater(tf.abs(self.input), 0), dtype=tf.float32)
        
        #print("mask shape: {}".format(self.mask.get_shape()))

        #self.l1_loss = tf.reduce_mean(tf.losses.absolute_difference(self.pred, self.input))
        self.l2_loss = tf.pow(self.pred - self.input, 2)
        #self.loss = tf.reduce_sum(tf.losses.absolute_difference(self.pred, self.fc_input, reduction=tf.losses.Reduction.NONE) * self.mask) / (tf.reduce_sum(self.mask)) + self.l1_loss
        #self.loss = tf.reduce_mean(self.l2_loss * self.mask)# + tf.reduce_mean(self.l2_loss)
        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.pred, self.input))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['train'])

            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("output_image",self.pred, collections=['train'])
            #tf.summary.image("code_image",self.code, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image",self.input , collections=['test'])
            tf.summary.image("output_image",self.pred, collections=['test'])
            #tf.summary.image("code_image",self.code, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline(self):
        """
        Training process.
        """     
        print("Training...")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        #mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
           
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 50
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            batch_images, batch_project = get_batch_data(96, 96, self.batch_size)
            #batch_images, _ = mnist.train.next_batch(self.batch_size)
            #batch_images = batch_images.reshape([-1,28,28,1])
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_project,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss = self.sess.run([self.merged_summary_train, 
                                          self.loss],
                                          feed_dict={
                                                        self.input: batch_project,
                                                        self.dropout: 1.,
                                                        self.lr:self.learning_rate 
                                                    })                                                                                                                   
                
#                test_images = get_batch_data(self.image_size, self.image_size, self.batch_size)
#                test_sum, test_loss = self.sess.run([self.merged_summary_test, 
#                                          self.l1_loss],                                          
#                                          feed_dict={
#                                                        self.input: test_images,
#                                                        self.dropout: 1.,
#                                                    })                             
                                                                     
                print("Iter: [{}], Train: [{}]".format((it+1), train_loss))       
                
                summary_writer.add_summary(train_sum, it)
#                summary_writer.add_summary(test_sum, it) 
    
    def build_baseline_v2(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.project_image_size, self.project_image_size, 1], name='images')
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        self.code, self.pred = mz.build_model()       
        
        self.l2_loss = tf.pow(self.pred - self.input, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.pred, self.input))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['train'])

            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("output_image",self.pred, collections=['train'])
            #tf.summary.image("code_image",self.code, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image",self.input , collections=['test'])
            tf.summary.image("output_image",self.pred, collections=['test'])
            #tf.summary.image("code_image",self.code, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline_v2(self):
        """
        Training process.
        """     
        print("Training...")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        #mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
           
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 50
        
        util = utils.utility(self.image_size, self.image_size)
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            batch_images, batch_project = util.get_batch_data(self.batch_size)
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_project,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss, train_pred = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_project,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                                                                                                                   
                
                if it % (save_it*10) == 0 and it != 0:
                    decode_output = util.projection_reverse(train_pred, self.batch_size)
                    
                    scipy.misc.imsave('./output/encode_input_{}.jpg'.format(it), batch_images[0].squeeze())
                    scipy.misc.imsave('./output/decode_output_{}.jpg'.format(it), decode_output[0].squeeze())                
                                                                     
                print("Iter: [{}], Train: [{}]".format((it+1), train_loss))       
                
                summary_writer.add_summary(train_sum, it)

    def build_baseline_v3(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.project_image_size, self.project_image_size, 1], name='images')
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        self.code, self.pred = mz.build_model()       
        
        self.l2_loss = tf.pow(self.pred - self.input, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.pred, self.input))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['train'])

            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("output_image",self.pred, collections=['train'])
            #tf.summary.image("code_image",self.code, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image",self.input , collections=['test'])
            tf.summary.image("output_image",self.pred, collections=['test'])
            #tf.summary.image("code_image",self.code, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline_v3(self):
        """
        Training process.
        """     
        print("Training...")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        #mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
           
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 50
        
        util = utils.utility(self.image_size, self.image_size)
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            batch_images, batch_project = util.get_batch_data(self.batch_size)
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_project,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss, train_pred = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_project,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                                                                                                                   
                
                if it % (save_it*10) == 0 and it != 0:
                    decode_output = util.projection_reverse(train_pred, self.batch_size)
                    
                    self.save_output(batch_images[0], decode_output[0], self.ckpt_name, it)
                    #scipy.misc.imsave('./output/encode_input_{}.jpg'.format(it), batch_images[0].squeeze())
                    #scipy.misc.imsave('./output/decode_output_{}.jpg'.format(it), decode_output[0].squeeze())                
                                                                     
                print("Iter: [{}], Train: [{}]".format((it+1), train_loss))       
                
                summary_writer.add_summary(train_sum, it)    
    
    def build_baseline_v4(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.project_image_size, self.project_image_size, 1], name='images')
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = AE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        # Encoder
        self.code = mz.build_model({"mode":"encoder", "image_size":self.project_image_size})       
        # Decoder
        self.pred = mz.build_model({"mode":"decoder", "image_size":self.project_image_size, "code":self.code})       
        
        #self.l1_loss = tf.losses.absolute_difference(self.pred, self.input, reduction=tf.losses.Reduction.NONE)
        self.l2_loss = tf.pow(self.pred - self.input, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.pred, self.input))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['train'])

            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image", self.input , collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['train']) # *** clip_value_max
            #tf.summary.image("code_image",self.code, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image", self.input , collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['test'])
            #tf.summary.image("code_image",self.code, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline_v4(self):
        """
        Training process.
        """     
        print("Training...")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        #mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
           
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 50
        
        util = utils.utility(self.image_size, self.image_size)
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            batch_images, batch_project = util.get_batch_data(self.batch_size)
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_project,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss, train_pred = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_project,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                                                                                                                   
                
                if it % (save_it*10) == 0 and it != 0:
                    decode_output = util.projection_reverse(train_pred, 1)
                    
                    self.save_output(batch_images[0], decode_output[0], self.ckpt_name, it)
                    #scipy.misc.imsave('./output/encode_input_{}.jpg'.format(it), batch_images[0].squeeze())
                    #scipy.misc.imsave('./output/decode_output_{}.jpg'.format(it), decode_output[0].squeeze())                
                                                                     
                print("Iter: [{}], Train: [{}]".format((it+1), train_loss))       
                
                summary_writer.add_summary(train_sum, it)        
                
    def build_baseline_v5(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.project_image_size, self.project_image_size, 1], name='images')
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = AE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        # Encoder
        self.code = mz.build_model({"mode":"encoder", "image_size":self.project_image_size})       
        # Decoder
        self.pred = mz.build_model({"mode":"decoder", "image_size":self.project_image_size, "code":self.code})       
        
        #self.l1_loss = tf.losses.absolute_difference(self.pred, self.input, reduction=tf.losses.Reduction.NONE)
        self.l2_loss = tf.pow(self.pred - self.input, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.pred, self.input))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['train'])

            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image", self.input , collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['train']) # *** clip_value_max
            tf.summary.image("code_image",self.code, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image", self.input , collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['test'])
            #tf.summary.image("code_image",self.code, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline_v5(self):
        """
        Training process.
        """     
        print("Training...")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        #mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
           
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 50
        
        util = utils.utility(self.image_size, self.image_size, self.project_image_size)
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            batch_images, batch_project = util.get_batch_data(self.batch_size)
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_project,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss, train_pred = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_project,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                                                                                                                   
                
                if it % (save_it*10) == 0 and it != 0:
                    decode_output = util.projection_reverse(train_pred, 1)
                    
                    self.save_output(batch_images[0], decode_output[0], self.ckpt_name, it)
                    #scipy.misc.imsave('./output/encode_input_{}.jpg'.format(it), batch_images[0].squeeze())
                    #scipy.misc.imsave('./output/decode_output_{}.jpg'.format(it), decode_output[0].squeeze())                
                                                                     
                print("Iter: [{}], Train: [{}]".format((it+1), train_loss))       
                
                summary_writer.add_summary(train_sum, it)                        
                
    def build_baseline_v5_flatten(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.project_image_size, self.project_image_size, 1], name='images')
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = AE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        # Encoder
        self.code = mz.build_model({"mode":"encoder", "image_size":self.project_image_size})       
        # Decoder
        self.pred = mz.build_model({"mode":"decoder", "image_size":self.project_image_size, "code":self.code})       
        
        self.code_image = tf.reshape(self.code, [tf.shape(self.input)[0], 64, 64, -1])
        
        #self.l1_loss = tf.losses.absolute_difference(self.pred, self.input, reduction=tf.losses.Reduction.NONE)
        self.l2_loss = tf.pow(self.pred - self.input, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.pred, self.input))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['train'])

            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image", self.input , collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['train']) # *** clip_value_max
            tf.summary.image("code_image",self.code_image, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image", self.input , collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['test'])
            #tf.summary.image("code_image",self.code, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline_v5_flatten(self):
        """
        Training process.
        """     
        print("Training...")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        #mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
           
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 1000
        minimum_loss = 1000
        
        util = utils.utility(self.image_size, self.image_size, self.project_image_size)
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            batch_images, batch_project = util.get_batch_data(self.batch_size)
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_project,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss, train_pred = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_project,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                                                                                                                   
                
                if it % (save_it*10) == 0 and it != 0:
                    decode_output = util.projection_reverse(train_pred, 1)
                    
                    self.save_output(batch_images[0], decode_output[0], self.ckpt_name, it)
                    #scipy.misc.imsave('./output/encode_input_{}.jpg'.format(it), batch_images[0].squeeze())
                    #scipy.misc.imsave('./output/decode_output_{}.jpg'.format(it), decode_output[0].squeeze())                

                if minimum_loss > train_loss:
                    minimum_loss = train_loss
                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, minimum_loss, it)
                                                                     
                print("Iter: [{}], Train: [{}]".format((it+1), train_loss))       
                
                summary_writer.add_summary(train_sum, it)        
                
    def build_baseline_v6_flatten(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.project_image_size[0], self.project_image_size[1], 1], name='images')
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = AE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        # Encoder
        self.code = mz.build_model({"model_list":self.model_list, "mode":"encoder", "image_size":[self.project_image_size[0], self.project_image_size[1]]})       
        # Decoder
        self.pred = mz.build_model({"model_list":self.model_list, "mode":"decoder", "image_size":[self.project_image_size[0], self.project_image_size[1]], "code":self.code})       
        
        self.code_image = tf.reshape(self.code, [tf.shape(self.input)[0], 32, 32, -1])
        
        #self.l1_loss = tf.losses.absolute_difference(self.pred, self.input, reduction=tf.losses.Reduction.NONE)
        self.l2_loss = tf.pow(self.pred - self.input, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.pred, self.input))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['train'])

            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image", self.input , collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['train']) # *** clip_value_max
            tf.summary.image("code_image",self.code_image, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image", self.input , collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['test'])
            tf.summary.image("code_image",self.code_image, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline_v6_flatten(self):
        """
        Training process.
        """     
        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        #mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
        
        print("Load Test data...")        
        batch_test_project = []
        util = utils.utility(self.image_size, self.image_size, self.project_image_size[0])
        input_data, label_data, image_names = self.load_image_data()
        for i_idx in range(len(input_data)):
            split_input, split_label, check_list, meta_data = self.split_image(input_data[i_idx], label_data[i_idx], image_names[i_idx], self.image_size)
            curr_test_project = util.projection(split_label, len(split_label))
            batch_test_project.append(curr_test_project)
        
        batch_test_project = np.array(batch_test_project)
        batch_test_project = batch_test_project.reshape(-1, self.project_image_size[0], self.project_image_size[1], 1)
        print(batch_test_project.shape)            
            
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
           
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 1000
        minimum_loss = 1000
        
        print("Training...")       
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            batch_images, batch_project = util.get_batch_data(self.batch_size)
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_project,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss, train_pred = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_project,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                              
                                                                                     
                idx = list(range(0, len(batch_test_project)))
                random.shuffle(idx)
                batch_data_idx = idx[:self.batch_size]
                test_sum, test_loss = \
                                                        self.sess.run([self.merged_summary_test, 
                                                        self.loss],
                                                        feed_dict={
                                                                    self.input: batch_test_project[batch_data_idx],
                                                                  })                                                   
                
                if it % (save_it*10) == 0 and it != 0:
                    decode_output = util.projection_reverse(train_pred, 1)
                    
                    self.save_output(batch_images[0], decode_output[0], self.ckpt_name, it)
                    #scipy.misc.imsave('./output/encode_input_{}.jpg'.format(it), batch_images[0].squeeze())
                    #scipy.misc.imsave('./output/decode_output_{}.jpg'.format(it), decode_output[0].squeeze())                

                if minimum_loss > train_loss:
                    minimum_loss = train_loss
                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, minimum_loss, it)
                                                                     
                print("Iter: [{}], Train: [{}], Test: [{}]".format((it+1), train_loss, test_loss))       
                
                summary_writer.add_summary(test_sum, it)           
                summary_writer.add_summary(train_sum, it)           
                
    def build_baseline_end2end(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], name='images')
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = AE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        # Encoder
        self.code = mz.build_model({"model_list":self.model_list, "mode":"encoder", "image_size":[self.image_size, self.image_size]})       
        # Decoder
        self.pred = mz.build_model({"model_list":self.model_list, "mode":"decoder", "image_size":[self.image_size, self.image_size], "code":self.code})       
        
        self.code_image = tf.reshape(self.code, [tf.shape(self.input)[0], 64, 64, -1])
        
        self.l2_loss = tf.pow(self.pred - self.input, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.pred, self.input))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['train'])

            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image", self.input , collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['train']) # *** clip_value_max
            tf.summary.image("code_image",self.code_image, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image", self.input , collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['test'])
            tf.summary.image("code_image",self.code_image, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline_end2end(self):
        """
        Training process.
        """     
        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        #mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
        
#        print("Load Test data...")        
#        input_data, label_data, image_names = self.load_image_data()
        util = utils.utility(self.image_size, self.image_size, self.project_image_size[0])
#        batch_test_project = np.array(label_data)
        
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
           
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 1000
        minimum_loss = 1000
        
        print("Training...")       
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            batch_images, _ = util.get_batch_data(self.batch_size, project_enable=False)
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_images,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss, train_pred = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_images,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                              
                                                                                     
#                idx = list(range(0, len(batch_test_project)))
#                random.shuffle(idx)
#                batch_data_idx = idx[:self.batch_size]
#                test_sum, test_loss = \
#                                                        self.sess.run([self.merged_summary_test, 
#                                                        self.loss],
#                                                        feed_dict={
#                                                                    self.input: batch_test_project[batch_data_idx],
#                                                                  })                                                   
                         
                if minimum_loss > train_loss:
                    minimum_loss = train_loss
                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, minimum_loss, it)
                                                                     
                print("Iter: [{}], Train: [{}]".format((it+1), train_loss))       
                        
                summary_writer.add_summary(train_sum, it)           
                                
    def build_baseline_end2end_2D(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], name='images')
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = AE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        # Encoder
        self.code = mz.build_model({"model_list":self.model_list, "mode":"encoder", "image_size":[self.image_size, self.image_size]})       
        # Decoder
        self.pred = mz.build_model({"model_list":self.model_list, "mode":"decoder", "image_size":[self.image_size, self.image_size], "code":self.code})       
        
        self.code_image = tf.slice(self.code, [0,0,0,0], [tf.shape(self.input)[0], 16, 16, 3], name="code_image")
        print("code_image: %s" % self.code_image.get_shape())
        
        self.l2_loss = tf.pow(self.pred - self.input, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.pred, self.input))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['train'])

            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image", self.input , collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['train']) # *** clip_value_max
            tf.summary.image("code_image", self.code_image, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image", self.input , collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['test'])
            tf.summary.image("code_image", self.code_image, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline_end2end_2D(self):
        """
        Training process.
        """     
        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        #mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
        
#        print("Load Test data...")        
#        input_data, label_data, image_names = self.load_image_data()
        util = utils.utility(self.image_size, self.image_size, self.project_image_size[0])
#        batch_test_project = np.array(label_data)
        
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
           
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 1000
        minimum_loss = 1000
        
        print("Training...")       
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            batch_images, _ = util.get_batch_data(self.batch_size, project_enable=False)
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_images,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss, train_pred = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_images,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                              
                                                                                     
#                idx = list(range(0, len(batch_test_project)))
#                random.shuffle(idx)
#                batch_data_idx = idx[:self.batch_size]
#                test_sum, test_loss = \
#                                                        self.sess.run([self.merged_summary_test, 
#                                                        self.loss],
#                                                        feed_dict={
#                                                                    self.input: batch_test_project[batch_data_idx],
#                                                                  })                                                   
                         
                if minimum_loss > train_loss:
                    minimum_loss = train_loss
                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, minimum_loss, it)
                                                                     
                print("Iter: [{}], Train: [{}]".format((it+1), train_loss))       
                        
                summary_writer.add_summary(train_sum, it)                    
                
    def build_baseline_end2end_2D_v2(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], name='images')
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = AE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        # Encoder
        self.code = mz.build_model({"model_list":self.model_list, "mode":"encoder", "image_size":[self.image_size, self.image_size]})       
        # Decoder
        self.pred = mz.build_model({"model_list":self.model_list, "mode":"decoder", "image_size":[self.image_size, self.image_size], "code":self.code})       
        
        #self.code_image = tf.reshape(self.code, [tf.shape(self.input)[0], 64, 64, 1])
        #self.code_image = tf.slice(self.code, [0,0,0,0], [tf.shape(self.input)[0], 16, 16, 3], name="code_image")
        self.code_image = tf.slice(self.code, [0,0,0,0], [tf.shape(self.input)[0], 8, 8, 3], name="code_image")
       
        print("code_image: %s" % self.code_image.get_shape())
        
        self.l2_loss = tf.pow(self.pred - self.input, 2)

        self.loss = tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
        
        MSE = tf.reduce_mean(tf.squared_difference(self.pred, self.input))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/MSE
        PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['train'])

            tf.summary.scalar("MSE", MSE, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image", self.input , collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['train']) # *** clip_value_max
            tf.summary.image("code_image", self.code_image, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("l1_loss", self.loss, collections=['test'])
            
            tf.summary.scalar("MSE", MSE, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image", self.input , collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred, 0, 500), collections=['test'])
            tf.summary.image("code_image", self.code_image, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline_end2end_2D_v2(self):
        """
        Training process.
        """     
        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        #mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
        
#        print("Load Test data...")        
#        input_data, label_data, image_names = self.load_image_data()
        util = utils.utility(self.image_size, self.image_size, self.project_image_size[0])
#        batch_test_project = np.array(label_data)
        
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
           
        print("Current learning rate: [{}]".format(self.learning_rate))
        
        save_it = 1000
        minimum_loss = 1000
        
        print("Training...")       
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            batch_images, _ = util.get_batch_data(self.batch_size, project_enable=False)
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_images,
                                           self.dropout: 1.,
                                           self.lr:self.learning_rate 
                                         })
                  
            if it % save_it == 0 and it != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, it)
                
                train_sum, train_loss, train_pred = \
                                                        self.sess.run([self.merged_summary_train, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_images,
                                                                    self.dropout: 1.,
                                                                    self.lr:self.learning_rate 
                                                                  })                              
                                                                                     
#                idx = list(range(0, len(batch_test_project)))
#                random.shuffle(idx)
#                batch_data_idx = idx[:self.batch_size]
#                test_sum, test_loss = \
#                                                        self.sess.run([self.merged_summary_test, 
#                                                        self.loss],
#                                                        feed_dict={
#                                                                    self.input: batch_test_project[batch_data_idx],
#                                                                  })                                                   
                         
                if minimum_loss > train_loss:
                    minimum_loss = train_loss
                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, minimum_loss, it)
                                                                     
                print("Iter: [{}], Train: [{}]".format((it+1), train_loss))       
                        
                summary_writer.add_summary(train_sum, it)                    