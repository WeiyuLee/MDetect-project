import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

import sys
sys.path.append('./utility')

def lrelu(x, name = "leaky", alpha = 0.2):

#    with tf.variable_scope(name):
#        leaky = tf.nn.relu(x) - alpha * tf.nn.relu(-x)
#    return leaky

    return tf.maximum(x, alpha * x, name=name)

#def lrelu(name,x, leak=0.2):
#    return tf.maximum(x, leak * x, name=name)

def convolution_layer(inputs, kernel_shape, stride, name, flatten = False ,padding = 'SAME',initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu, reg=None, is_bn=False):
                                                                                         
    pre_shape = inputs.get_shape()[-1]
    rkernel_shape = [kernel_shape[0], kernel_shape[1], pre_shape, kernel_shape[2]]     
    
    with tf.variable_scope(name) as scope:
        
        try:
            weight = tf.get_variable("weights",rkernel_shape, tf.float32, initializer=initializer, regularizer=reg)
            bias = tf.get_variable("bias",kernel_shape[2], tf.float32, initializer=tf.zeros_initializer())
        except:
            scope.reuse_variables()
            weight = tf.get_variable("weights",rkernel_shape, tf.float32, initializer=initializer, regularizer=reg)
            bias = tf.get_variable("bias",kernel_shape[2], tf.float32, initializer=tf.zeros_initializer())
        
        net = tf.nn.conv2d(inputs, weight,stride, padding=padding)
        net = tf.add(net, bias)
       
        if not activat_fn==None:
            net = activat_fn(net, name=name+"_out")
        
        if flatten == True:
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name=name+"_flatout")
        
    return net

def deconvolution_layer(inputs, kernel_shape, outshape, stride, name, flatten = False ,padding = 'SAME',initializer=tf.random_normal_initializer(stddev=0.01), activat_fn=lrelu, reg=None, is_bn=False):
                                                                                            
    pre_shape = inputs.get_shape()[-1]
    rkernel_shape = [kernel_shape[0], kernel_shape[1], kernel_shape[2], pre_shape]     
    
    with tf.variable_scope(name) as scope:
        
        try:
            weight = tf.get_variable("weights", rkernel_shape, tf.float32, initializer=initializer, regularizer=reg)
            bias = tf.get_variable("bias", kernel_shape[2], tf.float32, initializer=tf.zeros_initializer())
        except:
            scope.reuse_variables()
            weight = tf.get_variable("weights", rkernel_shape, tf.float32, initializer=initializer, regularizer=reg)
            bias = tf.get_variable("bias", kernel_shape[2], tf.float32, initializer=tf.zeros_initializer())
        
        net = tf.nn.conv2d_transpose(inputs, weight, outshape, stride, padding=padding)
        net = tf.add(net, bias)
       
        if not activat_fn==None:
            net = activat_fn(net, name=name+"_out")
        
        if flatten == True:
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name=name+"_flatout")
        
    return net


def max_pool_layer(inputs, kernel_shape, stride, name=None, padding='VALID'):
           
    return tf.nn.max_pool(inputs, kernel_shape, stride, padding, name=name)

def fc_layer(inputs, out_shape, name,initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu, reg=None):
    
    
    pre_shape = inputs.get_shape()[-1]
    
    with tf.variable_scope(name) as scope:
        
        
        try:
            weight = tf.get_variable("weights",[pre_shape, out_shape], tf.float32, initializer=initializer, regularizer=reg)
            bias = tf.get_variable("bias",out_shape, tf.float32, initializer=initializer)
        except:
            scope.reuse_variables()
            weight = tf.get_variable("weights",[pre_shape, out_shape], tf.float32, initializer=initializer, regularizer=reg)
            bias = tf.get_variable("bias",out_shape, tf.float32, initializer=initializer)
        
        
        if activat_fn != None:
            net = activat_fn(tf.nn.xw_plus_b(inputs, weight, bias, name=name + '_out'))
        else:
            net = tf.nn.xw_plus_b(inputs, weight, bias, name=name)
        
    return net

def inception_v1(inputs, module_shape, name, flatten = False ,padding = 'SAME',initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu):
    
    
    
    with tf.variable_scope(name):
        
            with tf.variable_scope("1x1"):
                
                kernel_shape = module_shape[name]["1x1"]
                net1x1 = convolution_layer(inputs, [1,1,kernel_shape], [1,1,1,1], name="conv1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
                
            with tf.variable_scope("3x3"):
                
                kernel_shape = module_shape[name]["3x3"]["1x1"]
                net3x3 = convolution_layer(inputs, [1,1,kernel_shape], [1,1,1,1], name="conv1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
                kernel_shape = module_shape[name]["3x3"]["3x3"]
                net3x3 = convolution_layer(net3x3, [3,3,kernel_shape], [1,1,1,1], name="conv3x3", padding=padding, initializer=initializer, activat_fn = activat_fn)
                
            with tf.variable_scope("5x5"):
                
                kernel_shape = module_shape[name]["5x5"]["1x1"]
                net5x5 = convolution_layer(inputs, [1,1,kernel_shape], [1,1,1,1], name="conv1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
                kernel_shape = module_shape[name]["5x5"]["5x5"]
                net5x5 = convolution_layer(net5x5, [5,5,kernel_shape], [1,1,1,1], name="conv5x5", padding=padding, initializer=initializer, activat_fn = activat_fn)
                
            with tf.variable_scope("s1x1"):
                            
                kernel_shape = module_shape[name]["s1x1"]
                net_s1x1 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1], padding=padding, name = "maxpool_s1x1")
                net_s1x1 = convolution_layer(net_s1x1, [1,1,kernel_shape], [1,1,1,1], name="conv_s1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
            
            net = tf.concat([net1x1, net3x3, net5x5, net_s1x1], axis=3)
            
            if flatten == True:
                net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name=name+"_flatout")
                
            
            return net

def shortcut(inputs, identity, name):  #Use 1X1 conv with proper stride to match dimesions
    
    in_shape =  inputs.get_shape().as_list()
    res_shape = identity.get_shape().as_list()
    
    dim_diff = [res_shape[1]/in_shape[1],
                res_shape[2]/in_shape[2]]
    
    
    if dim_diff[0] > 1  and dim_diff[1] > 1:
    
        identity = convolution_layer(identity, [1,1,in_shape[3]], [1,dim_diff[0],dim_diff[1],1], name="shotcut", padding="VALID")
    
    resout = tf.add(inputs, identity, name=name)
    
    return resout

def resBlock(x,channels=64,kernel_size=[3,3],scale=1, reuse = False, is_bn = False, idx = 0, initializer=tf.contrib.layers.xavier_initializer(),activation_fn=tf.nn.relu):

    tmp = slim.conv2d(x,channels,kernel_size,activation_fn=activation_fn, weights_initializer=initializer)
#    if is_bn:
#        tmp = batchnorm(tmp, index = idx, reuse = reuse)
#    else:
#        tmp = tmp 
    tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None, weights_initializer=initializer)
    tmp *= scale
    return x + tmp


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    