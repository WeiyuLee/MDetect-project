import tensorflow as tf
import netfactory as nf
import numpy as np

class model_zoo:
    
    def __init__(self, inputs, dropout, is_training, model_ticket):
        
        self.model_ticket = model_ticket
        self.inputs = inputs
        self.dropout = dropout
        self.is_training = is_training
        
    def googleLeNet_v1(self):
        
        model_params = {
        
            "conv1": [5,5, 64],
            "conv2": [3,3,128],
            "inception_1":{                 
                    "1x1":64,
                    "3x3":{ "1x1":96,
                            "3x3":128
                            },
                    "5x5":{ "1x1":16,
                            "5x5":32
                            },
                    "s1x1":32
                    },
            "inception_2":{                 
                    "1x1":128,
                    "3x3":{ "1x1":128,
                            "3x3":192
                            },
                    "5x5":{ "1x1":32,
                            "5x5":96
                            },
                    "s1x1":64
                    },
            "fc3": 10,
                     
        }
                
        
        with tf.name_scope("googleLeNet_v1"):
            net = nf.convolution_layer(self.inputs, model_params["conv1"], [1,2,2,1],name="conv1")
            net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.local_response_normalization(net, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='LocalResponseNormalization')
            net = nf.convolution_layer(net, model_params["conv2"], [1,1,1,1],name="conv2", flatten=False)
            net = tf.nn.local_response_normalization(net, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='LocalResponseNormalization')
            net = nf.inception_v1(net, model_params, name= "inception_1", flatten=False)
            net = nf.inception_v1(net, model_params, name= "inception_2", flatten=False)
            net = tf.nn.avg_pool (net, ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1], padding='VALID')
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))])
            
            net = tf.layers.dropout(net, rate=self.dropout, training=self.is_training, name='dropout2')
            logits = nf.fc_layer(net, model_params["fc3"], name="logits", activat_fn=None)

            
        return logits
        
    def resNet_v1(self):
        
        model_params = {
        
            "conv1": [5,5, 64],
            "rb1_1": [3,3,64],
            "rb1_2": [3,3,64],
            "rb2_1": [3,3,128],
            "rb2_2": [3,3,128],
            "fc3": 10,
                     
        }
                
        
        with tf.name_scope("resNet_v1"):
            net = nf.convolution_layer(self.inputs, model_params["conv1"], [1,2,2,1],name="conv1")
            id_rb1 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
            
            net = nf.convolution_layer(id_rb1, model_params["rb1_1"], [1,1,1,1],name="rb1_1")
            id_rb2 = nf.convolution_layer(net, model_params["rb1_2"], [1,1,1,1],name="rb1_2")
            
            id_rb2 = nf.shortcut(id_rb2,id_rb1, name="rb1")
            
            net = nf.convolution_layer(id_rb2, model_params["rb2_1"], [1,2,2,1],padding="SAME",name="rb2_1")
            id_rb3 = nf.convolution_layer(net, model_params["rb2_2"], [1,1,1,1],name="rb2_2")
            
            id_rb3 = nf.shortcut(id_rb3,id_rb2, name="rb2")
            
            net  = nf.global_avg_pooling(id_rb3, flatten=True)
            
            net = tf.layers.dropout(net, rate=self.dropout, training=self.is_training, name='dropout2')
            logits = nf.fc_layer(net, model_params["fc3"], name="logits", activat_fn=None)

            
        return logits

    def baseline(self):
        
        model_params = {
        
            "conv_1": [3,3,16],
            "conv_2": [3,3,32],

            "conv_3": [1,1,3],
                       
            "deconv_3": [1,1,32],
            
            "deconv_2": [3,3,16],
            "deconv_1": [3,3,1]
        }
                
        num_resblock = 16
        
        
        with tf.name_scope("baseline"):
            conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME')
            conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2", padding='SAME')
            
            with tf.variable_scope("encoder_resblock",reuse=False): 
                en_rb_x = conv_2
                #Add the residual blocks to the model
                for i in range(num_resblock):
                    en_rb_x = nf.resBlock(en_rb_x, 32, scale=1, reuse=False, idx = i, initializer=tf.random_normal_initializer(stddev=0.01))
                en_rb_x = nf.convolution_layer(en_rb_x, model_params["conv_2"], [1,1,1,1], name="conv_3", activat_fn=None, initializer=tf.random_normal_initializer(stddev=0.01))
                en_rb_x += conv_2
            
            code_layer = en_rb_x
            print("code layer shape : %s" % code_layer.get_shape())

            with tf.variable_scope("decoder_resblock",reuse=False): 
                de_rb_x = code_layer
                #Add the residual blocks to the model
                for i in range(num_resblock):
                    de_rb_x = nf.resBlock(de_rb_x, 32, scale=1, reuse=False, idx = i, initializer=tf.random_normal_initializer(stddev=0.01))
                de_rb_x = nf.convolution_layer(de_rb_x, model_params["conv_2"], [1,1,1,1], name="deconv_3", activat_fn=None, initializer=tf.random_normal_initializer(stddev=0.01))
                de_rb_x += code_layer            

            deconv_2 = self.deconv2d("deconv_2", de_rb_x, ksize=3, outshape=[tf.shape(self.inputs)[0], 97, 97, 16])
            deconv_2 = tf.nn.relu(deconv_2)
            deconv_1 = self.deconv2d("deconv_1", deconv_2, ksize=3, outshape=[tf.shape(self.inputs)[0], 193, 193, 1])
            deconv_1 = tf.nn.sigmoid(deconv_1)
                       
        return code_layer, deconv_1
    
    def baseline_v2(self):
        
        model_params = {
        
            "conv_1": [3,3,16],
            "conv_2": [3,3,32],

            "conv_3": [1,1,3],
                       
            "deconv_3": [1,1,32],
            
            "deconv_2": [3,3,16],
            "deconv_1": [3,3,1]
        }
                
        num_resblock = 16
        
        
        with tf.name_scope("baseline"):
            conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME')
            conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2", padding='SAME')
            
            with tf.variable_scope("encoder_resblock",reuse=False): 
                en_rb_x = conv_2
                #Add the residual blocks to the model
                for i in range(num_resblock):
                    en_rb_x = nf.resBlock(en_rb_x, 32, scale=1, reuse=False, idx = i, initializer=tf.random_normal_initializer(stddev=0.01))
                en_rb_x = nf.convolution_layer(en_rb_x, model_params["conv_2"], [1,1,1,1], name="conv_3", activat_fn=None, initializer=tf.random_normal_initializer(stddev=0.01))
                en_rb_x += conv_2
            
            code_layer = en_rb_x
            print("code layer shape : %s" % code_layer.get_shape())

            with tf.variable_scope("decoder_resblock",reuse=False): 
                de_rb_x = code_layer
                #Add the residual blocks to the model
                for i in range(num_resblock):
                    de_rb_x = nf.resBlock(de_rb_x, 32, scale=1, reuse=False, idx = i, initializer=tf.random_normal_initializer(stddev=0.01))
                de_rb_x = nf.convolution_layer(de_rb_x, model_params["conv_2"], [1,1,1,1], name="deconv_3", activat_fn=None, initializer=tf.random_normal_initializer(stddev=0.01))
                de_rb_x += code_layer            

            deconv_2 = self.deconv2d("deconv_2", de_rb_x, ksize=3, outshape=[tf.shape(self.inputs)[0], tf.shape(de_rb_x)[1]*2, tf.shape(de_rb_x)[2]*2, 16])
            deconv_2 = tf.nn.relu(deconv_2)
            deconv_1 = self.deconv2d("deconv_1", deconv_2, ksize=3, outshape=[tf.shape(self.inputs)[0], tf.shape(self.inputs)[1], tf.shape(self.inputs)[2], 1])
            deconv_1 = tf.nn.relu(deconv_1)
                       
        return code_layer, deconv_1    

    def baseline_v3(self):
        
        model_params = {
        
            "conv_1": [3,3,32],
            "conv_2": [3,3,16],
            "conv_3": [3,3,2],
                      
            "deconv_3": [1,1,32],            
            "deconv_2": [3,3,16],
            "deconv_1": [3,3,1]
        }
                
        num_resblock = 32
        
        with tf.name_scope("baseline"):
            conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME')
            conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2", padding='SAME')
            conv_3 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3", padding='SAME')
            
            with tf.variable_scope("encoder_resblock",reuse=False): 
                en_rb_x = conv_3
                #Add the residual blocks to the model
                for i in range(num_resblock):
                    en_rb_x = nf.resBlock(en_rb_x, model_params["conv_3"][2], scale=1, reuse=False, idx = i, initializer=tf.random_normal_initializer(stddev=0.01))
                en_rb_x = nf.convolution_layer(en_rb_x, model_params["conv_3"], [1,1,1,1], name="conv_4", activat_fn=None, initializer=tf.random_normal_initializer(stddev=0.01))
                en_rb_x += conv_3
            
            code_layer = en_rb_x
            print("code layer shape : %s" % code_layer.get_shape())

            with tf.variable_scope("decoder_resblock",reuse=False): 
                de_rb_x = code_layer
                #Add the residual blocks to the model
                for i in range(num_resblock):
                    de_rb_x = nf.resBlock(de_rb_x, model_params["conv_3"][2], scale=1, reuse=False, idx = i, initializer=tf.random_normal_initializer(stddev=0.01))
                de_rb_x = nf.convolution_layer(de_rb_x, model_params["conv_3"], [1,1,1,1], name="deconv_4", activat_fn=None, initializer=tf.random_normal_initializer(stddev=0.01))
                de_rb_x += code_layer            

            deconv_3 = self.deconv2d("deconv_3", de_rb_x, ksize=3, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_2)[1], tf.shape(conv_2)[1], 16])
            deconv_3 = tf.nn.relu(deconv_3)
            deconv_2 = self.deconv2d("deconv_2", deconv_3, ksize=3, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_1)[1], tf.shape(conv_1)[1], 32])
            deconv_2 = tf.nn.relu(deconv_2)
            deconv_1 = self.deconv2d("deconv_1", deconv_2, ksize=3, outshape=[tf.shape(self.inputs)[0], tf.shape(self.inputs)[1], tf.shape(self.inputs)[2], 1])
            deconv_1 = tf.nn.relu(deconv_1)
                       
        return code_layer, deconv_1 
    
#    def baseline_v4(self):
#         
#        model_params = {
#        
#            "conv_1": [3,3,32],
#            "conv_2": [3,3,16],
#            
#            "code": [3,3,4],
#            
#            "deconv_2": [3,3,16],
#            "deconv_1": [3,3,32]
#        }
#                
#        num_resblock = 16
#        
#        init = tf.random_normal_initializer(stddev=0.01)
#        
#        with tf.name_scope("baseline"):
#            conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME', initializer=init)
#            conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2", padding='SAME', initializer=init)
#            conv_3 = nf.convolution_layer(conv_2, model_params["code"], [1,2,2,1], name="conv_3", padding='SAME', initializer=init)
#            
#            with tf.variable_scope("encoder_resblock",reuse=False): 
#                en_rb_x = conv_3
#                #Add the residual blocks to the model
#                for i in range(num_resblock):
#                    en_rb_x = nf.resBlock(en_rb_x, model_params["code"][2], scale=1, reuse=False, idx = i, initializer=init)
#                en_rb_x = nf.convolution_layer(en_rb_x, model_params["code"], [1,1,1,1], name="conv_4", activat_fn=None, initializer=init)
#                en_rb_x += conv_3
#            
#            code_layer = en_rb_x
#            print("code layer shape : %s" % code_layer.get_shape())
#
#            with tf.variable_scope("decoder_resblock",reuse=False): 
#                de_rb_x = code_layer
#                #Add the residual blocks to the model
#                for i in range(num_resblock):
#                    de_rb_x = nf.resBlock(de_rb_x, model_params["code"][2], scale=1, reuse=False, idx = i, initializer=init)
#                de_rb_x = nf.convolution_layer(de_rb_x, model_params["code"], [1,1,1,1], name="deconv_4", activat_fn=None, initializer=init)
#                de_rb_x += code_layer            
#
#            deconv_3 = self.deconv2d("deconv_3", de_rb_x, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_2)[1], tf.shape(conv_2)[2], 16])
#            #deconv_3 = tf.nn.relu(deconv_3)
#            deconv_3 = nf.lrelu(deconv_3)
#            deconv_2 = self.deconv2d("deconv_2", deconv_3, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_1)[1], tf.shape(conv_1)[2], 32])
#            #deconv_2 = tf.nn.relu(deconv_2)            
#            deconv_2 = nf.lrelu(deconv_2)            
#            deconv_1 = self.deconv2d("deconv_1", deconv_2, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], tf.shape(self.inputs)[1], tf.shape(self.inputs)[2], 1])
#            #deconv_1 = tf.nn.relu(deconv_1)
#            deconv_1 = nf.lrelu(deconv_1)
#                       
#        return code_layer, deconv_1     
        
#    def baseline_v4(self, kwargs):
#         
#        model_params = {
#        
#            "conv_1": [3,3,32],
#            "conv_2": [3,3,16],
#            "conv_3": [3,3,8],
#            
#            "code": [3,3,4],
#            
#            "deconv_3": [3,3,8],
#            "deconv_2": [3,3,16],
#            "deconv_1": [3,3,32]
#        }
#
#        mode = kwargs["mode"]
#                
#        num_resblock = 16
#        
#        init = tf.random_normal_initializer(stddev=0.01)
#
#        if mode is "encoder":                
#            with tf.name_scope("encoder"):
#                conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME')
#                # 181x181x32
#                conv_1_1 = nf.convolution_layer(conv_1, model_params["conv_1"], [1,1,1,1], name="conv_1_1", padding='SAME')
#                conv_1_2 = nf.convolution_layer(conv_1 + conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2", padding='SAME')
#                conv_1_3 = nf.convolution_layer(conv_1 + conv_1_1 + conv_1_2, model_params["conv_1"], [1,1,1,1], name="conv_1_3", padding='SAME')
#                
#                conv_2 = nf.convolution_layer(conv_1 + conv_1_1 + conv_1_2 + conv_1_3, model_params["conv_2"], [1,2,2,1], name="conv_2", padding='SAME')
#                # 91x91x16
#                conv_2_1 = nf.convolution_layer(conv_2, model_params["conv_2"], [1,1,1,1], name="conv_2_1", padding='SAME')
#                conv_2_2 = nf.convolution_layer(conv_2 + conv_2_1, model_params["conv_2"], [1,1,1,1], name="conv_2_2", padding='SAME')
#                conv_2_3 = nf.convolution_layer(conv_2 + conv_2_1 + conv_2_2, model_params["conv_2"], [1,1,1,1], name="conv_2_3", padding='SAME')
#                
#                conv_3 = nf.convolution_layer(conv_2 + conv_2_1 + conv_2_2 + conv_2_3, model_params["conv_3"], [1,1,1,1], name="conv_3", padding='SAME')
#                # 91x91x8
#                conv_3_1 = nf.convolution_layer(conv_3, model_params["conv_3"], [1,1,1,1], name="conv_3_1", padding='SAME')
#                conv_3_2 = nf.convolution_layer(conv_3 + conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME')
#                conv_3_3 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME')
#                
#                conv_4 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2 + conv_3_3, model_params["code"], [1,2,2,1], name="conv_4", padding='SAME')
#                # 46x46x4
#                with tf.variable_scope("encoder_resblock",reuse=False): 
#                    en_rb_x = conv_4
#                    #Add the residual blocks to the model
#                    for i in range(num_resblock):
#                        en_rb_x = nf.resBlock(en_rb_x, model_params["code"][2], scale=1, reuse=False, idx = i, initializer=init)
#                    en_rb_x = nf.convolution_layer(en_rb_x, model_params["code"], [1,1,1,1], name="conv_5", activat_fn=None, initializer=init)
#                    en_rb_x += conv_4
#                    #en_rb_x = tf.nn.relu(en_rb_x)
#            
#                code_layer = en_rb_x
#                print("Encoder: code layer's shape is %s" % code_layer.get_shape())
#                
#            return code_layer
#
#        if mode is "decoder": 
#            
#            code_layer = kwargs["code"]
#            print("Decoder: code layer's shape is %s" % code_layer.get_shape())
#            
#            with tf.name_scope("decoder"):           
#    
#                with tf.variable_scope("decoder_resblock",reuse=False): 
#                    de_rb_x = code_layer
#                    #Add the residual blocks to the model
#                    for i in range(num_resblock):
#                        de_rb_x = nf.resBlock(de_rb_x, model_params["code"][2], scale=1, reuse=False, idx = i, initializer=init)
#                    de_rb_x = nf.convolution_layer(de_rb_x, model_params["code"], [1,1,1,1], name="deconv_5", activat_fn=None, initializer=init)
#                    de_rb_x += code_layer            
#                    #de_rb_x = tf.nn.relu(de_rb_x)
#    
#                deconv_4   = nf.lrelu(self.deconv2d("deconv_4", de_rb_x,                              ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_3)[1], tf.shape(conv_3)[2], 8]))            
#                # 91x91x8
#                deconv_3_3 = nf.lrelu(self.deconv2d("deconv_3_3", deconv_4,                           ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_3)[1], tf.shape(conv_3)[2], 8]))           
#                deconv_3_2 = nf.lrelu(self.deconv2d("deconv_3_2", deconv_3_3 + deconv_4,              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_3)[1], tf.shape(conv_3)[2], 8]))            
#                deconv_3_1 = nf.lrelu(self.deconv2d("deconv_3_1", deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_3)[1], tf.shape(conv_3)[2], 8]))            
#                
#                deconv_3   = nf.lrelu(self.deconv2d("deconv_3", deconv_3_1 + deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_2)[1], tf.shape(conv_2)[2], 16]))            
#                # 91x91x16
#                deconv_2_3 = nf.lrelu(self.deconv2d("deconv_2_3", deconv_3,                           ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_2)[1], tf.shape(conv_2)[2], 16]))      
#                deconv_2_2 = nf.lrelu(self.deconv2d("deconv_2_2", deconv_2_3 + deconv_3,              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_2)[1], tf.shape(conv_2)[2], 16]))      
#                deconv_2_1 = nf.lrelu(self.deconv2d("deconv_2_1", deconv_2_2 + deconv_2_3 + deconv_3, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_2)[1], tf.shape(conv_2)[2], 16]))      
#    
#                deconv_2   = nf.lrelu(self.deconv2d("deconv_2", deconv_2_1 + deconv_2_2 + deconv_2_3 + deconv_3, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_1)[1], tf.shape(conv_1)[2], 32]))            
#                # 181x181x32
#                deconv_1_3 = nf.lrelu(self.deconv2d("deconv_1_3", deconv_2,                           ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_1)[1], tf.shape(conv_1)[2], 32]))      
#                deconv_1_2 = nf.lrelu(self.deconv2d("deconv_1_2", deconv_1_3 + deconv_2,              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_1)[1], tf.shape(conv_1)[2], 32]))      
#                deconv_1_1 = nf.lrelu(self.deconv2d("deconv_1_1", deconv_1_2 + deconv_1_3 + deconv_2, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], tf.shape(conv_1)[1], tf.shape(conv_1)[2], 32]))      
#                
#                deconv_1   = nf.lrelu(self.deconv2d("deconv_1", deconv_1_1 + deconv_1_2 + deconv_1_3 + deconv_2, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], tf.shape(self.inputs)[1], tf.shape(self.inputs)[2], 1]))
#                       
#            return deconv_1       
        
    def baseline_v4(self, kwargs):
         
        model_params = {
        
            "conv_1": [3,3,64],
            "conv_2": [3,3,32],
            "conv_3": [3,3,16],
            "conv_4": [3,3,8],
            
            #"fc_encode": 32*32,
            "fc_encode": 64*64,
            "fc_decode": 46*46*8,
            
            "deconv_4": [3,3,8],
            "deconv_3": [3,3,16],
            "deconv_2": [3,3,32],
            "deconv_1": [3,3,64]
        }

        mode = kwargs["mode"]
        
        image_size = kwargs["image_size"]
                
        num_resblock = 16
        
        init = tf.random_normal_initializer(stddev=0.01)

        if mode is "encoder":                
            with tf.name_scope("encoder"):
                conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME')
                # 181x181x32
                print("conv_1: %s" % conv_1.get_shape())
                
                
                conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2", padding='SAME')
                # 91x91x16
                print("conv_2: %s" % conv_2.get_shape())
                
                conv_3 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,1,1,1], name="conv_3", padding='SAME')
                # 91x91x8
                print("conv_3: %s" % conv_3.get_shape())
                
                conv_3_1 = nf.convolution_layer(conv_3, model_params["conv_3"], [1,1,1,1], name="conv_3_1", padding='SAME')
                conv_3_2 = nf.convolution_layer(conv_3 + conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME')
                conv_3_3 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME')
                
                conv_4 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2 + conv_3_3, model_params["conv_4"], [1,2,2,1], name="conv_4", padding='SAME')
                # 46x46x4
                print("conv_4: %s" % conv_4.get_shape())
                
                with tf.variable_scope("encoder_resblock",reuse=False): 
                    en_rb_x = conv_4
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        en_rb_x = nf.resBlock(en_rb_x, model_params["conv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    en_rb_x = nf.convolution_layer(en_rb_x, model_params["conv_4"], [1,1,1,1], name="conv_5", activat_fn=None, initializer=init)
                    en_rb_x += conv_4
                    #en_rb_x = tf.nn.relu(en_rb_x)
                    
                en_rb_x_flatten = tf.reshape(en_rb_x, [tf.shape(self.inputs)[0], 46*46*8])
                fc_encode = nf.fc_layer(en_rb_x_flatten, model_params["fc_encode"], name="encode", activat_fn=tf.nn.relu)
                
                code_layer = fc_encode
                print("Encoder: code layer's shape is %s" % code_layer.get_shape())
                
            return code_layer

        if mode is "decoder": 
            
            code_layer = kwargs["code"]
            print("Decoder: code layer's shape is %s" % code_layer.get_shape())
            
            fc_decode = nf.fc_layer(code_layer, model_params["fc_decode"], name="decode", activat_fn=tf.nn.relu)
            decode_reshape = tf.reshape(fc_decode, [tf.shape(self.inputs)[0], 46, 46, 8])
            
            with tf.name_scope("decoder"):           
    
                with tf.variable_scope("decoder_resblock",reuse=False): 
                    de_rb_x = decode_reshape
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        de_rb_x = nf.resBlock(de_rb_x, model_params["deconv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    de_rb_x = nf.convolution_layer(de_rb_x, model_params["deconv_4"], [1,1,1,1], name="deconv_5", activat_fn=None, initializer=init)
                    de_rb_x += decode_reshape            
                    #de_rb_x = tf.nn.relu(de_rb_x)
    
                deconv_4   = nf.lrelu(self.deconv2d("deconv_4", de_rb_x,                              ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size//4+1, image_size//4+1, 16]))            
                # 91x91x8
                print("deconv_4: %s" % deconv_4.get_shape())
                
                deconv_3_3 = nf.lrelu(self.deconv2d("deconv_3_3", deconv_4,                           ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//4+1, image_size//4+1, 16]))           
                deconv_3_2 = nf.lrelu(self.deconv2d("deconv_3_2", deconv_3_3 + deconv_4,              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//4+1, image_size//4+1, 16]))            
                deconv_3_1 = nf.lrelu(self.deconv2d("deconv_3_1", deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//4+1, image_size//4+1, 16]))            
                
                deconv_3   = nf.lrelu(self.deconv2d("deconv_3", deconv_3_1 + deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//4+1, image_size//4+1, 32]))            
                # 91x91x16  
                print("deconv_3: %s" % deconv_3.get_shape())
                
                deconv_2   = nf.lrelu(self.deconv2d("deconv_2", deconv_3, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size//2+1, image_size//2+1, 64]))            
                # 182x182x32  
                print("deconv_2: %s" % deconv_2.get_shape())
                
                deconv_1   = nf.lrelu(self.deconv2d("deconv_1", deconv_2, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size, image_size, 1]))
                print("deconv_1: %s" % deconv_1.get_shape())
                
            return deconv_1           
        
    def baseline_v5(self, kwargs):
         
        model_params = {
        
            "conv_1": [3,3,64],
            "conv_2": [3,3,32],
            "conv_3": [5,5,16],
            
            "code": [3,3,4],
            
            "deconv_3": [5,5,16],
            "deconv_2": [3,3,32],
            "deconv_1": [3,3,64]
        }

        mode = kwargs["mode"]
        
        image_size = kwargs["image_size"]
                
        num_resblock = 16
        
        init = tf.random_normal_initializer(stddev=0.01)

        if mode is "encoder":                
            with tf.name_scope("encoder"):
                conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME')
                # 128x128x64
                print("conv_1: %s" % conv_1.get_shape())                
                
                conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,1,1,1], name="conv_2", padding='SAME')
                # 64x64x32
                print("conv_2: %s" % conv_2.get_shape())
                
                conv_3 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,1,1,1], name="conv_3", padding='SAME')
                # 64x64x16
                print("conv_3: %s" % conv_3.get_shape())
                
                conv_3_1 = nf.convolution_layer(conv_3, model_params["conv_3"], [1,1,1,1], name="conv_3_1", padding='SAME')
                conv_3_2 = nf.convolution_layer(conv_3 + conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME')
                conv_3_3 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME')
                
                conv_4 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2 + conv_3_3, model_params["code"], [1,3,3,1], name="conv_4", padding='SAME')
                # 64x64x4
                print("conv_4: %s" % conv_4.get_shape())
                
                with tf.variable_scope("encoder_resblock",reuse=False): 
                    en_rb_x = conv_4
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        en_rb_x = nf.resBlock(en_rb_x, model_params["code"][2], scale=1, reuse=False, idx = i, initializer=init)
                    en_rb_x = nf.convolution_layer(en_rb_x, model_params["code"], [1,1,1,1], name="conv_5", activat_fn=None, initializer=init)
                    en_rb_x += conv_4
                    en_rb_x = tf.nn.relu(en_rb_x)
                
                code_layer = en_rb_x
                print("Encoder: code layer's shape is %s" % code_layer.get_shape())
                
            return code_layer

        if mode is "decoder": 
            
            code_layer = kwargs["code"]
            print("Decoder: code layer's shape is %s" % code_layer.get_shape())
            
            with tf.name_scope("decoder"):           
    
                with tf.variable_scope("decoder_resblock",reuse=False): 
                    de_rb_x = code_layer
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        de_rb_x = nf.resBlock(de_rb_x, model_params["code"][2], scale=1, reuse=False, idx = i, initializer=init)
                    de_rb_x = nf.convolution_layer(de_rb_x, model_params["code"], [1,1,1,1], name="deconv_5", activat_fn=None, initializer=init)
                    de_rb_x += code_layer            
                    de_rb_x = tf.nn.relu(de_rb_x)
    
                deconv_4   = nf.lrelu(self.deconv2d("deconv_4", de_rb_x,                              ksize=5, stride=3, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 16]))            
                # 91x91x8
                print("deconv_4: %s" % deconv_4.get_shape())
                
                deconv_3_3 = nf.lrelu(self.deconv2d("deconv_3_3", deconv_4,                           ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 16]))           
                deconv_3_2 = nf.lrelu(self.deconv2d("deconv_3_2", deconv_3_3 + deconv_4,              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 16]))            
                deconv_3_1 = nf.lrelu(self.deconv2d("deconv_3_1", deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 16]))            
                
                deconv_3   = nf.lrelu(self.deconv2d("deconv_3", deconv_3_1 + deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 32]))            
                # 91x91x16  
                print("deconv_3: %s" % deconv_3.get_shape())
                
                deconv_2   = nf.lrelu(self.deconv2d("deconv_2", deconv_3, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 64]))            
                # 182x182x32  
                print("deconv_2: %s" % deconv_2.get_shape())
                
                deconv_1   = nf.lrelu(self.deconv2d("deconv_1", deconv_2, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size, image_size, 1]))
                print("deconv_1: %s" % deconv_1.get_shape())
                
            return deconv_1           

    def baseline_v5_flatten(self, kwargs):
         
        model_params = {
        
            "conv_1": [3,3,64],
            "conv_2": [3,3,32],
            "conv_3": [5,5,16],
            "conv_4": [3,3,4],
            
            "fc_5": 64*64*2,
            #"fc_code": 64*64*4,
            "fc_code": 64*64,
            "fc_6": 64*64*4,
            
            "deconv_4": [3,3,4],
            "deconv_3": [5,5,16],
            "deconv_2": [3,3,32],
            "deconv_1": [3,3,64]
        }

        mode = kwargs["mode"]
        
        image_size = kwargs["image_size"]
                
        num_resblock = 16
        
        init = tf.random_normal_initializer(stddev=0.01)

        if mode is "encoder":                
            with tf.name_scope("encoder"):
                conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME')
                # 128x128x64
                print("conv_1: %s" % conv_1.get_shape())                
                
                conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,1,1,1], name="conv_2", padding='SAME')
                # 64x64x32
                print("conv_2: %s" % conv_2.get_shape())
                
                conv_3 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,1,1,1], name="conv_3", padding='SAME')
                # 64x64x16
                print("conv_3: %s" % conv_3.get_shape())
                
                conv_3_1 = nf.convolution_layer(conv_3, model_params["conv_3"], [1,1,1,1], name="conv_3_1", padding='SAME')
                conv_3_2 = nf.convolution_layer(conv_3 + conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME')
                conv_3_3 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME')
                
                conv_4 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2 + conv_3_3, model_params["conv_4"], [1,3,3,1], name="conv_4", padding='SAME')
                # 64x64x4
                print("conv_4: %s" % conv_4.get_shape())
                
                with tf.variable_scope("encoder_resblock",reuse=False): 
                    en_rb_x = conv_4
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        en_rb_x = nf.resBlock(en_rb_x, model_params["conv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    en_rb_x = nf.convolution_layer(en_rb_x, model_params["conv_4"], [1,1,1,1], name="conv_5", activat_fn=None, initializer=init)
                    en_rb_x += conv_4
                    en_rb_x = tf.nn.relu(en_rb_x)
                
                    en_rb_x = tf.reshape(en_rb_x, [tf.shape(self.inputs)[0], 64*64*4])
                
                fc_5 = nf.fc_layer(en_rb_x, model_params["fc_code"], name="fc_5", activat_fn=tf.nn.relu)
                print("fc_5: %s" % fc_5.get_shape())
                
                code_layer = fc_5
                print("Encoder: code layer's shape is %s" % code_layer.get_shape())
                
            return code_layer

        if mode is "decoder": 
            
            code_layer = kwargs["code"]
            print("Decoder: code layer's shape is %s" % code_layer.get_shape())
            
            fc_7 = nf.fc_layer(code_layer, model_params["fc_6"], name="fc_7", activat_fn=tf.nn.relu)
            print("fc_7: %s" % fc_7.get_shape())
            
            fc_7 = tf.reshape(fc_7, [tf.shape(self.inputs)[0], image_size//6, image_size//6, 4])
            
            with tf.name_scope("decoder"):           
    
                with tf.variable_scope("decoder_resblock",reuse=False): 
                    de_rb_x = fc_7
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        de_rb_x = nf.resBlock(de_rb_x, model_params["conv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    de_rb_x = nf.convolution_layer(de_rb_x, model_params["conv_4"], [1,1,1,1], name="deconv_5", activat_fn=None, initializer=init)
                    de_rb_x += fc_7            
                    de_rb_x = tf.nn.relu(de_rb_x)
    
                deconv_4   = nf.lrelu(self.deconv2d("deconv_4", de_rb_x,                              ksize=5, stride=3, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 16]))            
                # 91x91x8
                print("deconv_4: %s" % deconv_4.get_shape())
                
                deconv_3_3 = nf.lrelu(self.deconv2d("deconv_3_3", deconv_4,                           ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 16]))           
                deconv_3_2 = nf.lrelu(self.deconv2d("deconv_3_2", deconv_3_3 + deconv_4,              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 16]))            
                deconv_3_1 = nf.lrelu(self.deconv2d("deconv_3_1", deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 16]))            
                
                deconv_3   = nf.lrelu(self.deconv2d("deconv_3", deconv_3_1 + deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 32]))            
                # 91x91x16  
                print("deconv_3: %s" % deconv_3.get_shape())
                
                deconv_2   = nf.lrelu(self.deconv2d("deconv_2", deconv_3, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size//2, image_size//2, 64]))            
                # 182x182x32  
                print("deconv_2: %s" % deconv_2.get_shape())
                
                deconv_1   = nf.lrelu(self.deconv2d("deconv_1", deconv_2, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size, image_size, 1]))
                print("deconv_1: %s" % deconv_1.get_shape())
                
            return deconv_1                 

    def baseline_v6_flatten(self, kwargs):
         
        model_params = {
        
            "conv_1": [11,11,64],
            "conv_2": [5,5,128],
            "conv_3": [3,3,128],
            "conv_4": [3,3,64],
            
            "fc_code": 4096,
            "fc_6": 8*91*64,
            
            "deconv_4": [3,3,64],
            "deconv_3": [3,3,128],
            "deconv_2": [5,5,128],
            "deconv_1": [11,11,64]
        }

        mode = kwargs["mode"]
        
        image_size = kwargs["image_size"]
                
        num_resblock = 16
        
        init = tf.random_normal_initializer(stddev=0.01)

        if mode is "encoder":                
            with tf.name_scope("encoder"):
                conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME')
                # 128x128x64
                print("conv_1: %s" % conv_1.get_shape())                
                
                conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,1,1,1], name="conv_2", padding='SAME')
                # 64x64x32
                print("conv_2: %s" % conv_2.get_shape())
                
                conv_3 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,1,1,1], name="conv_3", padding='SAME')
                # 64x64x16
                print("conv_3: %s" % conv_3.get_shape())
                
                conv_3_1 = nf.convolution_layer(conv_3, model_params["conv_3"], [1,1,1,1], name="conv_3_1", padding='SAME')
                conv_3_2 = nf.convolution_layer(conv_3 + conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME')
                conv_3_3 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME')
                
                conv_4 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2 + conv_3_3, model_params["conv_4"], [1,2,2,1], name="conv_4", padding='SAME')
                # 64x64x4
                print("conv_4: %s" % conv_4.get_shape())
                
                with tf.variable_scope("encoder_resblock",reuse=False): 
                    en_rb_x = conv_4
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        en_rb_x = nf.resBlock(en_rb_x, model_params["conv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    en_rb_x = nf.convolution_layer(en_rb_x, model_params["conv_4"], [1,1,1,1], name="conv_5", activat_fn=None, initializer=init)
                    en_rb_x += conv_4
                    en_rb_x = tf.nn.relu(en_rb_x)
                
                    en_rb_x = tf.reshape(en_rb_x, [tf.shape(self.inputs)[0], 8*91*64])
                
                fc_5 = nf.fc_layer(en_rb_x, model_params["fc_code"], name="fc_5", activat_fn=tf.nn.relu)
                print("fc_5: %s" % fc_5.get_shape())
                
                code_layer = fc_5
                print("Encoder: code layer's shape is %s" % code_layer.get_shape())
                
            return code_layer

        if mode is "decoder": 
            
            code_layer = kwargs["code"]
            print("Decoder: code layer's shape is %s" % code_layer.get_shape())
            
            fc_7 = nf.fc_layer(code_layer, model_params["fc_6"], name="fc_7", activat_fn=tf.nn.relu)
            print("fc_7: %s" % fc_7.get_shape())
            
            fc_7 = tf.reshape(fc_7, [tf.shape(self.inputs)[0], image_size[0]//4+1, image_size[1]//4+1, 64])
            
            with tf.name_scope("decoder"):           
    
                with tf.variable_scope("decoder_resblock",reuse=False): 
                    de_rb_x = fc_7
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        de_rb_x = nf.resBlock(de_rb_x, model_params["conv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    de_rb_x = nf.convolution_layer(de_rb_x, model_params["conv_4"], [1,1,1,1], name="deconv_5", activat_fn=None, initializer=init)
                    de_rb_x += fc_7            
                    de_rb_x = tf.nn.relu(de_rb_x)
    
                deconv_4   = nf.lrelu(self.deconv2d("deconv_4", de_rb_x,                              ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//2, image_size[1]//2+1, 128]))            
                # 91x91x8
                print("deconv_4: %s" % deconv_4.get_shape())
                
                deconv_3_3 = nf.lrelu(self.deconv2d("deconv_3_3", deconv_4,                           ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//2, image_size[1]//2+1, 128]))           
                deconv_3_2 = nf.lrelu(self.deconv2d("deconv_3_2", deconv_3_3 + deconv_4,              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//2, image_size[1]//2+1, 128]))            
                deconv_3_1 = nf.lrelu(self.deconv2d("deconv_3_1", deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//2, image_size[1]//2+1, 128]))            
                
                deconv_3   = nf.lrelu(self.deconv2d("deconv_3", deconv_3_1 + deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//2, image_size[1]//2+1, 128]))            
                # 91x91x16  
                print("deconv_3: %s" % deconv_3.get_shape())
                
                deconv_2   = nf.lrelu(self.deconv2d("deconv_2", deconv_3, ksize=5, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//2, image_size[1]//2+1, 64]))            
                # 182x182x32  
                print("deconv_2: %s" % deconv_2.get_shape())
                
                deconv_1   = nf.lrelu(self.deconv2d("deconv_1", deconv_2, ksize=11, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0], image_size[1], 1]))
                print("deconv_1: %s" % deconv_1.get_shape())
                
            return deconv_1  

    def baseline_end2end(self, kwargs):
         
        model_params = {
        
            "conv_1": [11,11,64],
            "conv_2": [5,5,128],
            "conv_3": [3,3,128],
            "conv_4": [3,3,64],
            
            "fc_code": 4096,
            "fc_6": 16*16*64,
            
            "deconv_4": [3,3,64],
            "deconv_3": [3,3,128],
            "deconv_2": [5,5,128],
            "deconv_1": [11,11,64]
        }

        mode = kwargs["mode"]
        
        image_size = kwargs["image_size"]
                
        num_resblock = 16
        
        init = tf.random_normal_initializer(stddev=0.01)

        if mode is "encoder":                
            with tf.name_scope("encoder"):
                print("input: %s" % self.inputs.get_shape())
                
                conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME')
                # 128x128x64
                print("conv_1: %s" % conv_1.get_shape())                
                
                conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2", padding='SAME')
                # 64x64x128
                print("conv_2: %s" % conv_2.get_shape())
                
                conv_3 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3", padding='SAME')
                # 32x32x128
                print("conv_3: %s" % conv_3.get_shape())
                
                conv_3_1 = nf.convolution_layer(conv_3, model_params["conv_3"], [1,1,1,1], name="conv_3_1", padding='SAME')
                conv_3_2 = nf.convolution_layer(conv_3 + conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME')
                conv_3_3 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME')
                
                conv_4 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2 + conv_3_3, model_params["conv_4"], [1,2,2,1], name="conv_4", padding='SAME')
                # 16x16x64
                print("conv_4: %s" % conv_4.get_shape())
                
                with tf.variable_scope("encoder_resblock",reuse=False): 
                    en_rb_x = conv_4
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        en_rb_x = nf.resBlock(en_rb_x, model_params["conv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    en_rb_x = nf.convolution_layer(en_rb_x, model_params["conv_4"], [1,1,1,1], name="conv_5", activat_fn=None, initializer=init)
                    en_rb_x += conv_4
                    en_rb_x = tf.nn.relu(en_rb_x)
                
                    en_rb_x = tf.reshape(en_rb_x, [tf.shape(self.inputs)[0], 16*16*64])
                
                fc_5 = nf.fc_layer(en_rb_x, model_params["fc_code"], name="fc_5", activat_fn=tf.nn.relu)
                print("fc_5: %s" % fc_5.get_shape())
                
                code_layer = fc_5
                print("Encoder: code layer's shape is %s" % code_layer.get_shape())
                
            return code_layer

        if mode is "decoder": 
            
            code_layer = kwargs["code"]
            print("Decoder: code layer's shape is %s" % code_layer.get_shape())
            
            fc_7 = nf.fc_layer(code_layer, model_params["fc_6"], name="fc_7", activat_fn=tf.nn.relu)           
            fc_7 = tf.reshape(fc_7, [tf.shape(self.inputs)[0], image_size[0]//16, image_size[1]//16, 64])
            print("fc_7: %s" % fc_7.get_shape())
            
            with tf.name_scope("decoder"):           
    
                with tf.variable_scope("decoder_resblock",reuse=False): 
                    de_rb_x = fc_7
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        de_rb_x = nf.resBlock(de_rb_x, model_params["conv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    de_rb_x = nf.convolution_layer(de_rb_x, model_params["conv_4"], [1,1,1,1], name="deconv_5", activat_fn=None, initializer=init)
                    de_rb_x += fc_7            
                    de_rb_x = tf.nn.relu(de_rb_x)
    
                deconv_4   = nf.lrelu(self.deconv2d("deconv_4", de_rb_x,                              ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))            
                # 91x91x8
                print("deconv_4: %s" % deconv_4.get_shape())
                
                deconv_3_3 = nf.lrelu(self.deconv2d("deconv_3_3", deconv_4,                           ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))           
                deconv_3_2 = nf.lrelu(self.deconv2d("deconv_3_2", deconv_3_3 + deconv_4,              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))            
                deconv_3_1 = nf.lrelu(self.deconv2d("deconv_3_1", deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))            
                
                deconv_3   = nf.lrelu(self.deconv2d("deconv_3", deconv_3_1 + deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//4, image_size[1]//4, 128]))            
                # 91x91x16  
                print("deconv_3: %s" % deconv_3.get_shape())
                
                deconv_2   = nf.lrelu(self.deconv2d("deconv_2", deconv_3, ksize=5, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//2, image_size[1]//2, 64]))            
                # 182x182x32  
                print("deconv_2: %s" % deconv_2.get_shape())
                
                deconv_1   = nf.lrelu(self.deconv2d("deconv_1", deconv_2, ksize=11, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0], image_size[1], 1]))
                print("output: %s" % deconv_1.get_shape())
                
            return deconv_1  

    def baseline_end2end_2D(self, kwargs):
         
        model_params = {
        
            "conv_1": [11,11,64],
            "conv_2": [5,5,128],
            "conv_3": [3,3,128],
            "conv_4": [3,3,64],

            "conv_code": [3,3,16],            
            
            "deconv_4": [3,3,64],
            "deconv_3": [3,3,128],
            "deconv_2": [5,5,128],
            "deconv_1": [11,11,64]
        }

        mode = kwargs["mode"]
        
        image_size = kwargs["image_size"]
                
        num_resblock = 16
        
        init = tf.random_normal_initializer(stddev=0.01)

        if mode is "encoder":                
            with tf.name_scope("encoder"):
                print("input: %s" % self.inputs.get_shape())
                
                conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,2,2,1], name="conv_1", padding='SAME')
                # 128x128x64
                print("conv_1: %s" % conv_1.get_shape())                
                
                conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2", padding='SAME')
                # 64x64x128
                print("conv_2: %s" % conv_2.get_shape())
                
                conv_3 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3", padding='SAME')
                # 32x32x128
                print("conv_3: %s" % conv_3.get_shape())
                
                conv_3_1 = nf.convolution_layer(conv_3, model_params["conv_3"], [1,1,1,1], name="conv_3_1", padding='SAME')
                conv_3_2 = nf.convolution_layer(conv_3 + conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME')
                conv_3_3 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2, model_params["conv_3"], [1,1,1,1], name="conv_3_3", padding='SAME')
                
                conv_4 = nf.convolution_layer(conv_3 + conv_3_1 + conv_3_2 + conv_3_3, model_params["conv_4"], [1,2,2,1], name="conv_4", padding='SAME')
                # 16x16x64
                print("conv_4: %s" % conv_4.get_shape())
                
                with tf.variable_scope("encoder_resblock",reuse=False): 
                    en_rb_x = conv_4
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        en_rb_x = nf.resBlock(en_rb_x, model_params["conv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    en_rb_x = nf.convolution_layer(en_rb_x, model_params["conv_4"], [1,1,1,1], name="conv_5", activat_fn=None, initializer=init)
                    en_rb_x += conv_4
                    en_rb_x = tf.nn.relu(en_rb_x)

                conv_5 = nf.convolution_layer(en_rb_x, model_params["conv_code"], [1,1,1,1], name="conv_5", padding='SAME')               
                print("conv_5: %s" % conv_5.get_shape())
                
                code_layer = conv_5
                print("Encoder: code layer's shape is %s" % code_layer.get_shape())
                
            return code_layer

        if mode is "decoder": 
            
            code_layer = kwargs["code"]
            print("Decoder: code layer's shape is %s" % code_layer.get_shape())
            
            deconv_5   = nf.lrelu(self.deconv2d("deconv_5", code_layer,                              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//16, image_size[1]//16, 64]))            
            print("deconv_5: %s" % deconv_5.get_shape())            
            
            with tf.name_scope("decoder"):           
    
                with tf.variable_scope("decoder_resblock",reuse=False): 
                    de_rb_x = deconv_5
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        de_rb_x = nf.resBlock(de_rb_x, model_params["conv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    de_rb_x = nf.convolution_layer(de_rb_x, model_params["conv_4"], [1,1,1,1], name="deconv_5", activat_fn=None, initializer=init)
                    de_rb_x += deconv_5            
                    de_rb_x = tf.nn.relu(de_rb_x)
    
                deconv_4   = nf.lrelu(self.deconv2d("deconv_4", de_rb_x,                              ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))            
                # 91x91x8
                print("deconv_4: %s" % deconv_4.get_shape())
                
                deconv_3_3 = nf.lrelu(self.deconv2d("deconv_3_3", deconv_4,                           ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))           
                deconv_3_2 = nf.lrelu(self.deconv2d("deconv_3_2", deconv_3_3 + deconv_4,              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))            
                deconv_3_1 = nf.lrelu(self.deconv2d("deconv_3_1", deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))            
                
                deconv_3   = nf.lrelu(self.deconv2d("deconv_3", deconv_3_1 + deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//4, image_size[1]//4, 128]))            
                # 91x91x16  
                print("deconv_3: %s" % deconv_3.get_shape())
                
                deconv_2   = nf.lrelu(self.deconv2d("deconv_2", deconv_3, ksize=5, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//2, image_size[1]//2, 64]))            
                # 182x182x32  
                print("deconv_2: %s" % deconv_2.get_shape())
                
                deconv_1   = nf.lrelu(self.deconv2d("deconv_1", deconv_2, ksize=11, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0], image_size[1], 1]))
                print("output: %s" % deconv_1.get_shape())
                
            return deconv_1  

    def baseline_end2end_2D_v2(self, kwargs):
         
        model_params = {
        
#            "conv_1": [11,11,4],
#            "conv_2": [5,5,8],
#
#            "fc_code": 4096,            

            "conv_1": [11,11,4],
            "conv_2": [5,5,8],
            "conv_3": [3,3,16],
            
            "conv_4": [3,3,16],
                  
            "deconv_4": [3,3,64],
            "deconv_3": [3,3,128],
            "deconv_2": [5,5,128],
            "deconv_1": [11,11,64]
        }

        mode = kwargs["mode"]
        
        image_size = kwargs["image_size"]
                
        num_resblock = 16
        
        init = tf.random_normal_initializer(stddev=0.01)

        if mode is "encoder":                
            with tf.name_scope("encoder"):
                print("input: %s" % self.inputs.get_shape())
                
#                conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,4,4,1], name="conv_1", padding='SAME')
#                # 128x128x64
#                print("conv_1: %s" % conv_1.get_shape())                
#                
#                conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2", padding='SAME', flatten=True)
#                # 64x64x128
#                print("conv_2: %s" % conv_2.get_shape())
#
#                fc_code = nf.fc_layer(conv_2, model_params["fc_code"], name="fc_code", activat_fn=tf.nn.relu)
#                print("fc_code: %s" % fc_code.get_shape())

                conv_1 = nf.convolution_layer(self.inputs, model_params["conv_1"], [1,4,4,1], name="conv_1", padding='SAME', activat_fn=None)
                # 128x128x64
                print("conv_1: %s" % conv_1.get_shape())                
                
                conv_2 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2", padding='SAME', activat_fn=None)
                # 64x64x128
                print("conv_2: %s" % conv_2.get_shape())

                conv_3 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,2,2,1], name="conv_3", padding='SAME', activat_fn=None)
                # 64x64x128
                print("conv_3: %s" % conv_3.get_shape())

                code_layer = conv_3
                print("Encoder: code layer's shape is %s" % code_layer.get_shape())

#                conv_4 = nf.convolution_layer(conv_3, model_params["conv_4"], [1,2,2,1], name="conv_4", padding='SAME', activat_fn=None)
#                # 64x64x128
#                print("conv_4: %s" % conv_4.get_shape())
#                
#                code_layer = conv_4
#                print("Encoder: code layer's shape is %s" % code_layer.get_shape())
                
            return code_layer

        if mode is "decoder": 
            
            code_layer = kwargs["code"]
            print("Decoder: code layer's shape is %s" % code_layer.get_shape())
            
            #code_layer = tf.reshape(code_layer, [tf.shape(self.inputs)[0], 16, 16, 16])
            
            #deconv_5   = nf.lrelu(self.deconv2d("deconv_5", code_layer,                              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//16, image_size[1]//16, 64]))                        
            deconv_5   = nf.lrelu(self.deconv2d("deconv_5", code_layer,                              ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//16, image_size[1]//16, 64]))            
            print("deconv_5: %s" % deconv_5.get_shape())            
            
            with tf.name_scope("decoder"):           
    
                with tf.variable_scope("decoder_resblock",reuse=False): 
                    de_rb_x = deconv_5
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        de_rb_x = nf.resBlock(de_rb_x, model_params["deconv_4"][2], scale=1, reuse=False, idx = i, initializer=init)
                    de_rb_x = nf.convolution_layer(de_rb_x, model_params["deconv_4"], [1,1,1,1], name="deconv_5", activat_fn=None, initializer=init)
                    de_rb_x += deconv_5            
                    de_rb_x = tf.nn.relu(de_rb_x)
    
                deconv_4   = nf.lrelu(self.deconv2d("deconv_4", de_rb_x,                              ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))            
                # 91x91x8
                print("deconv_4: %s" % deconv_4.get_shape())
                
                deconv_3_3 = nf.lrelu(self.deconv2d("deconv_3_3", deconv_4,                           ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))           
                deconv_3_2 = nf.lrelu(self.deconv2d("deconv_3_2", deconv_3_3 + deconv_4,              ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))            
                deconv_3_1 = nf.lrelu(self.deconv2d("deconv_3_1", deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=1, outshape=[tf.shape(self.inputs)[0], image_size[0]//8, image_size[1]//8, 128]))            
                
                deconv_3   = nf.lrelu(self.deconv2d("deconv_3", deconv_3_1 + deconv_3_2 + deconv_3_3 + deconv_4, ksize=3, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//4, image_size[1]//4, 128]))            
                # 91x91x16  
                print("deconv_3: %s" % deconv_3.get_shape())
                
                deconv_2   = nf.lrelu(self.deconv2d("deconv_2", deconv_3, ksize=5, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0]//2, image_size[1]//2, 64]))            
                # 182x182x32  
                print("deconv_2: %s" % deconv_2.get_shape())
                
                deconv_1   = nf.lrelu(self.deconv2d("deconv_1", deconv_2, ksize=11, stride=2, outshape=[tf.shape(self.inputs)[0], image_size[0], image_size[1], 1]))
                print("output: %s" % deconv_1.get_shape())
                
            return deconv_1  

    def build_model(self, kwargs = {}):

        #model_list = ["googleLeNet_v1", "resNet_v1", "baseline", "baseline_v2", "baseline_v3", "baseline_v4", "baseline_v5", "baseline_v5_flatten", "baseline_v6_flatten"]
        
        model_list = kwargs["model_list"]
        
        if self.model_ticket not in model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
           
            fn = getattr(self,self.model_ticket)
            
            if kwargs == {}:
                netowrk = fn()
            else:
                netowrk = fn(kwargs)
            return netowrk
        
    def conv2d(self, name, tensor,ksize, out_dim, stddev=0.01, stride=2, padding='SAME'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [ksize, ksize, tensor.get_shape()[-1],out_dim], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev))
            var = tf.nn.conv2d(tensor,w,[1,stride, stride,1],padding=padding)
            b = tf.get_variable('b', [out_dim], 'float32',initializer=tf.constant_initializer(0.01))
            return tf.nn.bias_add(var, b)
    
    def deconv2d(self, name, tensor, ksize, outshape, stddev=0.01, stride=2, padding='SAME'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [ksize, ksize, outshape[-1], tensor.get_shape()[-1]], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev))
            var = tf.nn.conv2d_transpose(tensor, w, outshape, strides=[1, stride, stride, 1], padding=padding)
            b = tf.get_variable('b', [outshape[-1]], 'float32', initializer=tf.constant_initializer(0.01))
            return tf.nn.bias_add(var, b)        

    def fully_connected(self, name,value, output_shape):
        with tf.variable_scope(name, reuse=None) as scope:
            shape = value.get_shape().as_list()
            w = tf.get_variable('w', [shape[1], output_shape], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('b', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    
            return tf.matmul(value, w) + b
        
def unit_test():

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
    is_training = tf.placeholder(tf.bool, name='is_training')
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    mz = model_zoo(x, dropout, is_training,"resNet_v1")
    return mz.build_model()
    

#m = unit_test()