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
                 train_root=None,
                 test_root=None,
                 code_root=None,
                 train_code_root=None,
                 test_code_root=None):                 
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

        self.train_code_root = train_code_root
        self.test_code_root = test_code_root
               
        self.train_root = train_root
        self.test_root = test_root
        
        self.model_ticket = model_ticket
        
        self.model_list = ["baseline", "alex_net", "alex_net_2D", "baseline_2D"]
               
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

    def load_data(self, data_root):
        
        batch_num = 5
        
        n_image = []
        n_code = []
        ab_image = []
        ab_code = []
        
        print(data_root)
        
        for b_num in range(batch_num):
        
            print("Loading Batch [{}]...".format(b_num))
            
            f = open(os.path.join(data_root, "normal_data_{}.npy".format(b_num)), "rb")
            curr_n_image = np.array(np.load(f))
            curr_n_code = np.array(np.load(f))
            f.close()
                    
            f = open(os.path.join(data_root, "abnormal_data_{}.npy".format(b_num)), "rb")
            curr_ab_image = np.array(np.load(f))
            curr_ab_code = np.array(np.load(f))
            f.close()        
            
            if n_image == []:
                n_image = curr_n_image
                n_code = curr_n_code
                ab_image = curr_ab_image
                ab_code = curr_ab_code
            else:
                n_image = np.concatenate((n_image, curr_n_image))
                n_code = np.concatenate((n_code, curr_n_code))
                ab_image = np.concatenate((ab_image, curr_ab_image))
                ab_code = np.concatenate((ab_code, curr_ab_code))
            
            #n_image.append(curr_n_image)
            #n_code.append(curr_n_code)
            #ab_image.append(curr_ab_image)
            #ab_code.append(curr_ab_code)

        #n_image = np.array(n_image)
        #n_code = np.array(n_code)
        #ab_image = np.array(ab_image)
        #ab_code = np.array(ab_code)

#        print(n_image.shape)

#        n_image = n_image.reshape(-1, n_image.shape[2], n_image.shape[3], n_image.shape[4])
#        n_code = n_code.reshape(-1, n_code.shape[2])
#        ab_image = ab_image.reshape(-1, ab_image.shape[2], ab_image.shape[3], ab_image.shape[4])
#        ab_code = ab_code.reshape(-1, ab_code.shape[2])

        print("Normal image: {}".format(n_image.shape))
        print("Abnormal image: {}".format(ab_image.shape))

        return n_image, n_code, ab_image, ab_code

    def load_mean_image(self, data_root):
        
        f = open(os.path.join(data_root, "mean_image.pkl"), "rb")
        mean_image = pickle.load(f)
        f.close()
        
        return mean_image

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
        batch_data_idx = idx[:int(batch_size)]
        rest_idx = idx[int(batch_size):]
        
        return batch_data_idx, rest_idx
                
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

        train_n_image, train_n_code, train_ab_image, train_ab_code = self.load_data(self.train_root)
        
        test_n_image, test_n_code, test_ab_image, test_ab_code = self.load_data(self.test_root)
                
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
        minimum_loss = 1000
        n_ratio = 0.5
        ab_ratio = 0.5
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

#            n_idx = self.get_batch_data_idx(len(train_n_image), round(self.batch_size * n_ratio))
#            ab_idx = self.get_batch_data_idx(len(train_ab_image), round(self.batch_size * ab_ratio))
#
#            batch_train_data = np.concatenate((train_ab_image[ab_idx], train_n_image[n_idx]))
#            batch_train_label_data = np.concatenate((train_ab_code[ab_idx], train_n_code[n_idx]))

            ab_idx = self.get_batch_data_idx(len(train_ab_image), round(self.batch_size))

            batch_train_data = train_ab_image[ab_idx]
            batch_train_label_data = train_ab_code[ab_idx]
            
            # Normalization
            #batch_train_data = (batch_train_data - self.image_mean) / self.image_std
            #batch_train_label_data = (batch_train_label_data - self.code_mean) / self.code_std
            batch_train_data = batch_train_data
            batch_train_label_data = batch_train_label_data
            
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

#                n_idx = self.get_batch_data_idx(len(test_n_image), round(self.batch_size * 0.9))
#                ab_idx = self.get_batch_data_idx(len(test_ab_image), round(self.batch_size * 0.1))
#    
#                batch_test_data = np.concatenate((test_ab_image[ab_idx], test_n_image[n_idx]))
#                batch_test_label_data = np.concatenate((test_ab_code[ab_idx], test_n_code[n_idx]))

                ab_idx = self.get_batch_data_idx(len(test_ab_image), round(self.batch_size))
    
                batch_test_data = test_ab_image[ab_idx]
                batch_test_label_data = test_ab_code[ab_idx]
                
                # Normalization
                #batch_test_data = (batch_test_data - self.image_mean) / self.image_std
                #batch_test_label_data = (batch_test_label_data - self.code_mean) / self.code_std
                batch_test_data = batch_test_data
                batch_test_label_data = batch_test_label_data
                
                test_sum, test_loss, test_pred = \
                                                        self.sess.run([self.merged_summary_test, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_test_data,
                                                                    self.label: batch_test_label_data,
                                                                  })                    
                
#                if it % (5000) == 0 and it != 0:
#                    n_idx = self.get_batch_data_idx(len(test_n_image), self.batch_size)
#                    ab_idx = self.get_batch_data_idx(len(test_ab_image), self.batch_size)      
#                    
#                    batch_test_data = test_ab_image[ab_idx]
#                    batch_test_label_data = test_ab_code[ab_idx]
#                    
#                    batch_test_data = batch_test_data
#                    batch_test_label_data = batch_test_label_data
#                    
#                    test_ab_loss= \
#                                                            self.sess.run([self.loss],
#                                                            feed_dict={
#                                                                        self.input: batch_test_data,
#                                                                        self.label: batch_test_label_data,
#                                                                      })                    
#             
#                    batch_test_data = test_n_image[n_idx]
#                    batch_test_label_data = test_n_code[n_idx]
#                    
#                    batch_test_data = batch_test_data
#                    batch_test_label_data = batch_test_label_data
#                    
#                    test_n_loss= \
#                                                            self.sess.run([self.loss],
#                                                            feed_dict={
#                                                                        self.input: batch_test_data,
#                                                                        self.label: batch_test_label_data,
#                                                                      })                    
#
#                    #ab_ratio = test_ab_loss[0] / (test_ab_loss[0] + test_n_loss[0])
#                    #n_ratio = test_n_loss[0] / (test_ab_loss[0] + test_n_loss[0])
#    
#                    print("*** Iter: [{}], Test [ab, n] loss: [{}, {}], ratio [ab, n]: [{}, {}]".format((it+1), test_ab_loss, 
#                                                                                                          test_n_loss, 
#                                                                                                          test_ab_loss[0] / (test_ab_loss[0] + test_n_loss[0]), 
#                                                                                                          test_n_loss[0] / (test_ab_loss[0] + test_n_loss[0])))       
#                                                                        
##                    decode_output = util.projection_reverse(test_pred, 1)                   
##                    self.save_output(batch_train_data[0], decode_output[0], self.ckpt_name, it)

                if minimum_loss > test_loss:
                    minimum_loss = test_loss
                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, minimum_loss, it)
                                                                     
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

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)
        
        self.l2_loss = tf.pow(self.pred - self.label, 2)

#        self.loss = tf.reduce_mean(self.l2_loss) + self.reg_set_l2_loss
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

        train_n_image, train_n_code, train_ab_image, train_ab_code = self.load_data(self.train_root)
        
        val_idx, train_idx = self.get_batch_data_idx(len(train_ab_image), len(train_ab_image)*0.2)
        
        test_ab_image = train_ab_image[val_idx]
        test_ab_code = train_ab_code[val_idx]
        train_ab_image = train_ab_image[train_idx]
        train_ab_code = train_ab_code[train_idx]
        
        #test_n_image, test_n_code, test_ab_image, test_ab_code = self.load_data(self.test_root)
                       
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
        minimum_loss = 1000
        n_ratio = 0.5
        ab_ratio = 0.5
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

#            n_idx = self.get_batch_data_idx(len(train_n_image), round(self.batch_size * n_ratio))
#            ab_idx = self.get_batch_data_idx(len(train_ab_image), round(self.batch_size * ab_ratio))
#
#            batch_train_data = np.concatenate((train_ab_image[ab_idx], train_n_image[n_idx]))
#            batch_train_label_data = np.concatenate((train_ab_code[ab_idx], train_n_code[n_idx]))

            ab_idx, _ = self.get_batch_data_idx(len(train_ab_image), round(self.batch_size))

            batch_train_data = train_ab_image[ab_idx]
            batch_train_label_data = train_ab_code[ab_idx]
            
            # Normalization
            #batch_train_data = (batch_train_data - self.image_mean) / self.image_std
            #batch_train_label_data = (batch_train_label_data - self.code_mean) / self.code_std
            batch_train_data = batch_train_data
            batch_train_label_data = batch_train_label_data
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_train_data,
                                           self.label: batch_train_label_data,
                                           self.dropout: 0.0,
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
                                                                    self.dropout: 0.0,
                                                                    self.lr:self.learning_rate 
                                                                  })                                                                                                                   

#                n_idx = self.get_batch_data_idx(len(test_n_image), round(self.batch_size * 0.9))
#                ab_idx = self.get_batch_data_idx(len(test_ab_image), round(self.batch_size * 0.1))
#    
#                batch_test_data = np.concatenate((test_ab_image[ab_idx], test_n_image[n_idx]))
#                batch_test_label_data = np.concatenate((test_ab_code[ab_idx], test_n_code[n_idx]))

                ab_idx, _ = self.get_batch_data_idx(len(test_ab_image), round(self.batch_size))
    
                batch_test_data = test_ab_image[ab_idx]
                batch_test_label_data = test_ab_code[ab_idx]
                
                # Normalization
                #batch_test_data = (batch_test_data - self.image_mean) / self.image_std
                #batch_test_label_data = (batch_test_label_data - self.code_mean) / self.code_std
                batch_test_data = batch_test_data
                batch_test_label_data = batch_test_label_data
                
                test_sum, test_loss, test_pred = \
                                                        self.sess.run([self.merged_summary_test, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_test_data,
                                                                    self.label: batch_test_label_data,
                                                                    self.dropout: 0.0,
                                                                  })                    
                
#                if it % (5000) == 0 and it != 0:
#                    n_idx = self.get_batch_data_idx(len(test_n_image), self.batch_size)
#                    ab_idx = self.get_batch_data_idx(len(test_ab_image), self.batch_size)      
#                    
#                    batch_test_data = test_ab_image[ab_idx]
#                    batch_test_label_data = test_ab_code[ab_idx]
#                    
#                    batch_test_data = batch_test_data
#                    batch_test_label_data = batch_test_label_data
#                    
#                    test_ab_loss= \
#                                                            self.sess.run([self.loss],
#                                                            feed_dict={
#                                                                        self.input: batch_test_data,
#                                                                        self.label: batch_test_label_data,
#                                                                      })                    
#             
#                    batch_test_data = test_n_image[n_idx]
#                    batch_test_label_data = test_n_code[n_idx]
#                    
#                    batch_test_data = batch_test_data
#                    batch_test_label_data = batch_test_label_data
#                    
#                    test_n_loss= \
#                                                            self.sess.run([self.loss],
#                                                            feed_dict={
#                                                                        self.input: batch_test_data,
#                                                                        self.label: batch_test_label_data,
#                                                                      })                    
#
#                    #ab_ratio = test_ab_loss[0] / (test_ab_loss[0] + test_n_loss[0])
#                    #n_ratio = test_n_loss[0] / (test_ab_loss[0] + test_n_loss[0])
#    
#                    print("*** Iter: [{}], Test [ab, n] loss: [{}, {}], ratio [ab, n]: [{}, {}]".format((it+1), test_ab_loss, 
#                                                                                                          test_n_loss, 
#                                                                                                          test_ab_loss[0] / (test_ab_loss[0] + test_n_loss[0]), 
#                                                                                                          test_n_loss[0] / (test_ab_loss[0] + test_n_loss[0])))       
                                                                        
#                    decode_output = util.projection_reverse(test_pred, 1)                   
#                    self.save_output(batch_train_data[0], decode_output[0], self.ckpt_name, it)

                if minimum_loss > test_loss:
                    minimum_loss = test_loss
                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, minimum_loss, it)
                                                                     
                print("Iter: [{}], Train: [{}], Test: [{}]".format((it+1), train_loss, test_loss))       
                
                summary_writer.add_summary(test_sum, it)        
                summary_writer.add_summary(train_sum, it)                     
                
    def build_alex_net_2D(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='input')
        self.label = tf.placeholder(tf.float32, [None, self.code_size[0], self.code_size[1], self.code_size[2]], name='label')
        #self.label = tf.placeholder(tf.float32, [None, self.code_size[0]*self.code_size[1]*self.code_size[2]], name='label')
        self.code_image = tf.reshape(self.label, [tf.shape(self.label)[0], self.code_size[0], self.code_size[1], self.code_size[2]])
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = DE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        self.pred = mz.build_model()       

        self.code_image_top = tf.slice(self.code_image, [0,0,0,0], [tf.shape(self.input)[0], 16, 16, 3], name="code_image_top")
        self.pred_top = tf.slice(self.pred, [0,0,0,0], [tf.shape(self.input)[0], 16, 16, 3], name="pred_top")
        #self.code_image_top = self.code_image
        #self.pred_top = self.pred

        #self.label_2D = tf.reshape(self.label, [tf.shape(self.label)[0], self.code_size[0], self.code_size[1], self.code_size[2]])

        print("======================")
        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        print("======================")
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)
        
        #self.l2_loss = tf.pow(self.pred - self.label, 2)
        #self.l1_loss = tf.losses.absolute_difference(self.pred, self.label)
        self.l2_loss = tf.pow(self.pred - self.label, 2)
        self.l1_loss = tf.losses.absolute_difference(self.pred, self.label)
        
#        self.loss = tf.reduce_mean(self.l2_loss) + self.reg_set_l2_loss
#        self.loss = tf.reduce_mean(self.l2_loss)
        self.loss = tf.reduce_mean(self.l1_loss) + 1000 * tf.reduce_mean(self.l2_loss)

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
#        self.train_op = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(self.loss)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.loss, collections=['train'])
            tf.summary.image("input_image", self.input, collections=['train'])
            tf.summary.image("code_image", self.code_image_top, collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred_top, 0, 500), collections=['train'])
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.loss, collections=['test'])
            tf.summary.image("input_image", self.input, collections=['test'])
            tf.summary.image("code_image", self.code_image_top, collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred_top, 0, 500), collections=['test'])
            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_alex_net_2D(self):
        """
        Training process.
        """     
        print("Loading data...")

        train_n_image, train_n_code, train_ab_image, train_ab_code = self.load_data(self.train_root)

        train_mean_image = self.load_mean_image(self.train_code_root)
        train_n_image = train_n_image - train_mean_image
        train_ab_image = train_ab_image - train_mean_image
      
        train_min = np.min([np.min(train_n_image), np.min(train_ab_image)])   
        train_n_image = train_n_image - train_min
        train_ab_image = train_ab_image - train_min
        
        train_max = np.max([np.max(train_n_image), np.max(train_ab_image)])
        train_n_image = train_n_image / train_max * 255
        train_ab_image = train_ab_image / train_max * 255       
        
        print("Train data [Min:Max] = [{},{}]".format(np.min([np.min(train_n_image), np.min(train_ab_image)]), np.max([np.max(train_n_image), np.max(train_ab_image)])))
        
        #val_idx, train_idx = self.get_batch_data_idx(len(train_ab_image), len(train_ab_image)*0.2)
        
        #test_ab_image = train_ab_image[val_idx]
        #test_ab_code = train_ab_code[val_idx]
        #train_ab_image = train_ab_image[train_idx]
        #train_ab_code = train_ab_code[train_idx]
        
        test_n_image, test_n_code, test_ab_image, test_ab_code = self.load_data(self.test_root)

        test_mean_image = self.load_mean_image(self.test_code_root)
        test_n_image = test_n_image - test_mean_image
        test_ab_image = test_ab_image - test_mean_image
        
        #test_min = np.min([np.min(test_n_image), np.min(test_ab_image)])
        test_n_image = test_n_image - train_min
        test_ab_image = test_ab_image - train_min

        #test_max = np.max([np.max(test_n_image), np.max(test_ab_image)])
        test_n_image = test_n_image / train_max * 255
        test_ab_image = test_ab_image / train_max * 255      
        
        print("Test data [Min:Max] = [{},{}]".format(np.min([np.min(test_n_image), np.min(test_ab_image)]), np.max([np.max(test_n_image), np.max(test_ab_image)])))
               
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
        minimum_loss = 1000
        n_ratio = 0.5
        ab_ratio = 0.5
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

#            n_idx, _ = self.get_batch_data_idx(len(train_n_image), round(self.batch_size * n_ratio))
#            ab_idx, _ = self.get_batch_data_idx(len(train_ab_image), round(self.batch_size * ab_ratio))
#
#            batch_train_data = np.concatenate((train_ab_image[ab_idx], train_n_image[n_idx]))
#            batch_train_label_data = np.concatenate((train_ab_code[ab_idx], train_n_code[n_idx]))

            ab_idx, _ = self.get_batch_data_idx(len(train_ab_image), round(self.batch_size))

            batch_train_data = train_ab_image[ab_idx]
            batch_train_label_data = train_ab_code[ab_idx] + 4.505359172821045
            
            # Normalization
            #batch_train_data = (batch_train_data - self.image_mean) / self.image_std
            #batch_train_label_data = (batch_train_label_data - self.code_mean) / self.code_std
            batch_train_data = batch_train_data
            batch_train_label_data = batch_train_label_data
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_train_data,
                                           self.label: batch_train_label_data,
                                           self.dropout: 0.0,
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
                                                                    self.dropout: 0.0,
                                                                    self.lr:self.learning_rate 
                                                                  })                                                                                                                   

#                n_idx, _ = self.get_batch_data_idx(len(test_n_image), round(self.batch_size * n_ratio))
#                ab_idx, _ = self.get_batch_data_idx(len(test_ab_image), round(self.batch_size * ab_ratio))
#    
#                batch_test_data = np.concatenate((test_ab_image[ab_idx], test_n_image[n_idx]))
#                batch_test_label_data = np.concatenate((test_ab_code[ab_idx], test_n_code[n_idx]))

                ab_idx, _ = self.get_batch_data_idx(len(test_ab_image), round(self.batch_size))
    
                batch_test_data = test_ab_image[ab_idx]
                batch_test_label_data = test_ab_code[ab_idx] + 4.505359172821045
                
                # Normalization
                #batch_test_data = (batch_test_data - self.image_mean) / self.image_std
                #batch_test_label_data = (batch_test_label_data - self.code_mean) / self.code_std
                batch_test_data = batch_test_data
                batch_test_label_data = batch_test_label_data
                
                test_sum, test_loss, test_pred = \
                                                        self.sess.run([self.merged_summary_test, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_test_data,
                                                                    self.label: batch_test_label_data,
                                                                    self.dropout: 0.0
                                                                  })                    
                
#                if it % (5000) == 0 and it != 0:
#                    n_idx, _ = self.get_batch_data_idx(len(test_n_image), self.batch_size)
#                    ab_idx, _ = self.get_batch_data_idx(len(test_ab_image), self.batch_size)      
#                    
#                    batch_test_data = test_ab_image[ab_idx]
#                    batch_test_label_data = test_ab_code[ab_idx]
#                    
#                    batch_test_data = batch_test_data
#                    batch_test_label_data = batch_test_label_data
#                    
#                    test_ab_loss= \
#                                                            self.sess.run([self.loss],
#                                                            feed_dict={
#                                                                        self.input: batch_test_data,
#                                                                        self.label: batch_test_label_data,
#                                                                        self.dropout: 0.0
#                                                                      })                    
#             
#                    batch_test_data = test_n_image[n_idx]
#                    batch_test_label_data = test_n_code[n_idx]
#                    
#                    batch_test_data = batch_test_data
#                    batch_test_label_data = batch_test_label_data
#                    
#                    test_n_loss= \
#                                                            self.sess.run([self.loss],
#                                                            feed_dict={
#                                                                        self.input: batch_test_data,
#                                                                        self.label: batch_test_label_data,
#                                                                        self.dropout: 0.0
#                                                                      })                    
#
#                    #ab_ratio = test_ab_loss[0] / (test_ab_loss[0] + test_n_loss[0])
#                    #n_ratio = test_n_loss[0] / (test_ab_loss[0] + test_n_loss[0])
#    
#                    print("*** Iter: [{}], Test [ab, n] loss: [{}, {}], ratio [ab, n]: [{}, {}]".format((it+1), test_ab_loss, 
#                                                                                                          test_n_loss, 
#                                                                                                          test_ab_loss[0] / (test_ab_loss[0] + test_n_loss[0]), 
#                                                                                                          test_n_loss[0] / (test_ab_loss[0] + test_n_loss[0])))       
                                                                        
#                    decode_output = util.projection_reverse(test_pred, 1)                   
#                    self.save_output(batch_train_data[0], decode_output[0], self.ckpt_name, it)

                if minimum_loss > test_loss:
                    minimum_loss = test_loss
                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, minimum_loss, it)
                                                                     
                print("Iter: [{}], Train: [{}], Test: [{}]".format((it+1), train_loss, test_loss))       
                
                summary_writer.add_summary(test_sum, it)        
                summary_writer.add_summary(train_sum, it)                                     
                
    def build_baseline_2D(self):###
        """
        Build Baseline model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='input')
        self.label = tf.placeholder(tf.float32, [None, self.code_size[0], self.code_size[1], self.code_size[2]], name='label')
        self.code_image = tf.reshape(self.label, [tf.shape(self.label)[0], self.code_size[0], self.code_size[1], self.code_size[2]])
        
        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       
        # Initial model_zoo
        mz = DE_model_zoo.model_zoo(self.input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        self.pred = mz.build_model()       

        self.code_image_top = tf.slice(self.code_image, [0,0,0,0], [tf.shape(self.input)[0], 16, 16, 3], name="code_image_top")
        self.pred_top = tf.slice(self.pred, [0,0,0,0], [tf.shape(self.input)[0], 16, 16, 3], name="pred_top")

#        print("======================")
#        print("Regular Set:")
#        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#        for key in keys:
#            print(key.name)
#        print("======================")
#        
#        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#        self.reg_set_l2_loss = tf.add_n(self.reg_set)
        
        self.l2_loss = tf.pow(self.pred - self.label, 2)
        self.l1_loss = tf.losses.absolute_difference(self.pred, self.label)

#        self.loss = tf.reduce_mean(self.l2_loss) + self.reg_set_l2_loss
#        self.loss = tf.reduce_mean(self.l2_loss)
#        self.loss = tf.reduce_mean(self.l1_loss)        
        self.loss = tf.reduce_mean(self.l1_loss) + tf.reduce_mean(self.l2_loss)             

        self.train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss)
#        self.train_op = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(self.loss)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.loss, collections=['train'])
            tf.summary.image("input_image", self.input, collections=['train'])
            tf.summary.image("code_image", self.code_image_top, collections=['train'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred_top, 0, 500), collections=['train'])
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.loss, collections=['test'])
            tf.summary.image("input_image", self.input, collections=['test'])
            tf.summary.image("code_image", self.code_image_top, collections=['test'])
            tf.summary.image("output_image", tf.clip_by_value(self.pred_top, 0, 500), collections=['test'])
            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_baseline_2D(self):
        """
        Training process.
        """     
        print("Loading data...")

        train_n_image, train_n_code, train_ab_image, train_ab_code = self.load_data(self.train_root)

        train_mean_image = self.load_mean_image(self.train_code_root)
        train_n_image = train_n_image - train_mean_image
        train_ab_image = train_ab_image - train_mean_image
      
        train_min = np.min([np.min(train_n_image), np.min(train_ab_image)])   
        train_n_image = train_n_image - train_min
        train_ab_image = train_ab_image - train_min
        
        train_max = np.max([np.max(train_n_image), np.max(train_ab_image)])
        train_n_image = train_n_image / train_max * 255
        train_ab_image = train_ab_image / train_max * 255       
        
        print("Train data [Min:Max] = [{},{}]".format(np.min([np.min(train_n_image), np.min(train_ab_image)]), np.max([np.max(train_n_image), np.max(train_ab_image)])))
        
        #val_idx, train_idx = self.get_batch_data_idx(len(train_ab_image), len(train_ab_image)*0.2)
        
        #test_ab_image = train_ab_image[val_idx]
        #test_ab_code = train_ab_code[val_idx]
        #train_ab_image = train_ab_image[train_idx]
        #train_ab_code = train_ab_code[train_idx]
        
        test_n_image, test_n_code, test_ab_image, test_ab_code = self.load_data(self.test_root)

        test_mean_image = self.load_mean_image(self.test_code_root)
        test_n_image = test_n_image - test_mean_image
        test_ab_image = test_ab_image - test_mean_image
        
        #test_min = np.min([np.min(test_n_image), np.min(test_ab_image)])
        test_n_image = test_n_image - train_min
        test_ab_image = test_ab_image - train_min

        #test_max = np.max([np.max(test_n_image), np.max(test_ab_image)])
        test_n_image = test_n_image / train_max * 255
        test_ab_image = test_ab_image / train_max * 255      
        
        print("Test data [Min:Max] = [{},{}]".format(np.min([np.min(test_n_image), np.min(test_ab_image)]), np.max([np.max(test_n_image), np.max(test_ab_image)])))
               
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
        minimum_loss = 1000
        n_ratio = 0.5
        ab_ratio = 0.5
        
        iter_pbar = tqdm(range(self.curr_iteration, self.iteration))
        for it in iter_pbar:            

            iter_pbar.set_description("Iter: [%2d], lr:%f" % ((it+1), self.learning_rate))

            n_idx, _ = self.get_batch_data_idx(len(train_n_image), round(self.batch_size * n_ratio))
            ab_idx, _ = self.get_batch_data_idx(len(train_ab_image), round(self.batch_size * ab_ratio))

            batch_train_data = np.concatenate((train_ab_image[ab_idx], train_n_image[n_idx]))
            batch_train_label_data = np.concatenate((train_ab_code[ab_idx], train_n_code[n_idx]))

#            ab_idx, _ = self.get_batch_data_idx(len(train_ab_image), round(self.batch_size))
#
#            batch_train_data = train_ab_image[ab_idx]
#            batch_train_label_data = train_ab_code[ab_idx]
            
            # Normalization
            #batch_train_data = (batch_train_data - self.image_mean) / self.image_std
            #batch_train_label_data = (batch_train_label_data - self.code_mean) / self.code_std
            batch_train_data = batch_train_data
            batch_train_label_data = batch_train_label_data
            
            _ = self.sess.run([self.train_op], 
                              feed_dict={
                                           self.input: batch_train_data,
                                           self.label: batch_train_label_data,
                                           self.dropout: 0.0,
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
                                                                    self.dropout: 0.0,
                                                                    self.lr:self.learning_rate 
                                                                  })                                                                                                                   

                n_idx, _ = self.get_batch_data_idx(len(test_n_image), round(self.batch_size * n_ratio))
                ab_idx, _ = self.get_batch_data_idx(len(test_ab_image), round(self.batch_size * ab_ratio))
    
                batch_test_data = np.concatenate((test_ab_image[ab_idx], test_n_image[n_idx]))
                batch_test_label_data = np.concatenate((test_ab_code[ab_idx], test_n_code[n_idx]))

#                ab_idx, _ = self.get_batch_data_idx(len(test_ab_image), round(self.batch_size))
#    
#                batch_test_data = test_ab_image[ab_idx]
#                batch_test_label_data = test_ab_code[ab_idx]
                
                # Normalization
                #batch_test_data = (batch_test_data - self.image_mean) / self.image_std
                #batch_test_label_data = (batch_test_label_data - self.code_mean) / self.code_std
                batch_test_data = batch_test_data
                batch_test_label_data = batch_test_label_data
                
                test_sum, test_loss, test_pred = \
                                                        self.sess.run([self.merged_summary_test, 
                                                        self.loss,
                                                        self.pred],
                                                        feed_dict={
                                                                    self.input: batch_test_data,
                                                                    self.label: batch_test_label_data,
                                                                    self.dropout: 0.0
                                                                  })                    
                
                if it % (5000) == 0 and it != 0:
                    n_idx, _ = self.get_batch_data_idx(len(test_n_image), self.batch_size)
                    ab_idx, _ = self.get_batch_data_idx(len(test_ab_image), self.batch_size)      
                    
                    batch_test_data = test_ab_image[ab_idx]
                    batch_test_label_data = test_ab_code[ab_idx]
                    
                    batch_test_data = batch_test_data
                    batch_test_label_data = batch_test_label_data
                    
                    test_ab_loss= \
                                                            self.sess.run([self.loss],
                                                            feed_dict={
                                                                        self.input: batch_test_data,
                                                                        self.label: batch_test_label_data,
                                                                        self.dropout: 0.0
                                                                      })                    
             
                    batch_test_data = test_n_image[n_idx]
                    batch_test_label_data = test_n_code[n_idx]
                    
                    batch_test_data = batch_test_data
                    batch_test_label_data = batch_test_label_data
                    
                    test_n_loss= \
                                                            self.sess.run([self.loss],
                                                            feed_dict={
                                                                        self.input: batch_test_data,
                                                                        self.label: batch_test_label_data,
                                                                        self.dropout: 0.0
                                                                      })                    

                    #ab_ratio = test_ab_loss[0] / (test_ab_loss[0] + test_n_loss[0])
                    #n_ratio = test_n_loss[0] / (test_ab_loss[0] + test_n_loss[0])
    
                    print("*** Iter: [{}], Test [ab, n] loss: [{}, {}], ratio [ab, n]: [{}, {}]".format((it+1), test_ab_loss, 
                                                                                                          test_n_loss, 
                                                                                                          test_ab_loss[0] / (test_ab_loss[0] + test_n_loss[0]), 
                                                                                                          test_n_loss[0] / (test_ab_loss[0] + test_n_loss[0])))       
                                                                        
#                    decode_output = util.projection_reverse(test_pred, 1)                   
#                    self.save_output(batch_train_data[0], decode_output[0], self.ckpt_name, it)

                if minimum_loss > test_loss:
                    minimum_loss = test_loss
                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, minimum_loss, it)
                                                                     
                print("Iter: [{}], Train: [{}], Test: [{}]".format((it+1), train_loss, test_loss))       
                
                summary_writer.add_summary(test_sum, it)        
                summary_writer.add_summary(train_sum, it)                                                     