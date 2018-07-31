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
       
    def baseline(self, kwargs = {}):
        init = tf.random_normal_initializer(stddev=0.01)

        feature_size = 64
       
        model_params = {

                        'conv1': [11,11,feature_size*2],
                        'conv2': [5,5,feature_size*4],
                        'resblock': [3,3,feature_size*4],
                        'conv3': [3,3,feature_size*4],
                        'fc4': 64*64*4,
                        'fc5': 64*64*2,
                        'fc_code': 1024,

                        }

        ### Generator
        num_resblock = 16
                   
        g_input = self.inputs
        
        with tf.name_scope("Detector"):  
            # 256x256x1
            x = nf.convolution_layer(g_input, model_params["conv1"], [1,2,2,1], name="conv1", activat_fn=tf.nn.relu, initializer=init)
            conv_1 = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
            print("conv_1: %s" % conv_1.get_shape())

            x = nf.convolution_layer(conv_1, model_params["conv2"], [1,2,2,1], name="conv2", activat_fn=tf.nn.relu, initializer=init)
            conv_2 = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
            print("conv_2: %s" % conv_2.get_shape())
            
            x = conv_2
            # 128x128xfeature_size
            with tf.variable_scope("detector_resblock",reuse=False):            
                #Add the residual blocks to the model
                for i in range(num_resblock):
                    x = nf.resBlock(x, feature_size*4, scale=1, reuse=False, idx = i, initializer=init)
                x = nf.convolution_layer(x, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=tf.nn.relu, initializer=init)
                x += conv_2
                print("conv_3: %s" % x.get_shape())
                
            x = nf.convolution_layer(x, model_params["conv3"], [1,2,2,1], name="conv4",  activat_fn=tf.nn.relu, initializer=init, flatten=True)
            print("conv_4: %s" % x.get_shape())
            
            fc_4 = nf.fc_layer(x, model_params["fc4"], name="fc_4", activat_fn=tf.nn.relu)
            print("fc_4: %s" % fc_4.get_shape())
            
            fc_5 = nf.fc_layer(fc_4, model_params["fc5"], name="fc_5", activat_fn=tf.nn.relu)
            print("fc_5: %s" % fc_5.get_shape())
            
            fc_code = nf.fc_layer(fc_5, model_params["fc_code"], name="fc_code", activat_fn=None)
            print("fc_code: %s" % fc_code.get_shape())
            
        return fc_code         

    def baseline_2D(self, kwargs = {}):
        init = tf.random_normal_initializer(stddev=0.01)

        feature_size = 64
       
        model_params = {

                        'conv1': [11,11,feature_size*2],
                        'conv2': [5,5,feature_size*4],
                        'resblock': [3,3,feature_size*4],
                        'conv3': [3,3,feature_size*4],
                        'conv4': [3,3,feature_size*8],
                        'conv5': [3,3,feature_size*8],                        
                        'conv6': [3,3,feature_size*4],
                        'conv_code': [3,3,16],

                        }

        ### Generator
        num_resblock = 16
                   
        g_input = self.inputs
        
        with tf.name_scope("Detector"):  
            # 256x256x1
            x = nf.convolution_layer(g_input, model_params["conv1"], [1,2,2,1], name="conv1", activat_fn=tf.nn.relu, initializer=init)
            conv_1 = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
            print("conv_1: %s" % conv_1.get_shape())

            x = nf.convolution_layer(conv_1, model_params["conv2"], [1,2,2,1], name="conv2", activat_fn=tf.nn.relu, initializer=init)
            conv_2 = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
            print("conv_2: %s" % conv_2.get_shape())
            
            x = conv_2
            # 128x128xfeature_size
            with tf.variable_scope("detector_resblock",reuse=False):            
                #Add the residual blocks to the model
                for i in range(num_resblock):
                    x = nf.resBlock(x, feature_size*4, scale=1, reuse=False, idx = i, initializer=init)
                x = nf.convolution_layer(x, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=tf.nn.relu, initializer=init)
                x += conv_2
                print("conv_3: %s" % x.get_shape())
                
            x = nf.convolution_layer(x, model_params["conv4"], [1,1,1,1], name="conv4",  activat_fn=tf.nn.relu, initializer=init)
            print("conv_4: %s" % x.get_shape())
            
            x = nf.convolution_layer(x, model_params["conv5"], [1,1,1,1], name="conv5",  activat_fn=tf.nn.relu, initializer=init)
            print("conv_5: %s" % x.get_shape())

            x = nf.convolution_layer(x, model_params["conv6"], [1,1,1,1], name="conv6",  activat_fn=tf.nn.relu, initializer=init)
            print("conv_6: %s" % x.get_shape())

            x = nf.convolution_layer(x, model_params["conv_code"], [1,1,1,1], name="conv_code",  activat_fn=tf.nn.relu, initializer=init)
            print("conv_code: %s" % x.get_shape())
            
        return x  

    def alex_net(self, kwargs = {}):

        init = tf.random_normal_initializer(stddev=0.01)
        l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        
        model_params = {

                        'conv1': [11,11,96],
                        'conv2': [5,5,256],
                        'conv3': [3,3,384],
                        'conv4': [3,3,384],
                        'conv5': [3,3,256],          
                        'fc6': 8192,                                  
                        'fc7': 8192,                                                          
                        'fc_code': 4096,            
#                        'fc7': 4096,                                                          
#                        'fc_code': 1024,            

                        
                        }        
        with tf.name_scope("Detector"):  
            
            conv_1 = nf.convolution_layer(self.inputs, model_params["conv1"], [1,4,4,1], name="conv1", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='VALID')
            conv_1 = tf.nn.max_pool(conv_1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
            
            conv_2 = nf.convolution_layer(conv_1, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')
            conv_2 = tf.nn.max_pool(conv_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

            conv_3 = nf.convolution_layer(conv_2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')

            conv_4 = nf.convolution_layer(conv_3, model_params["conv4"], [1,1,1,1], name="conv4", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')

            conv_5 = nf.convolution_layer(conv_4, model_params["conv5"], [1,1,1,1], name="conv5", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')
            conv_5 = tf.nn.max_pool(conv_5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
            conv_5 = tf.reshape(conv_5, [-1, int(np.prod(conv_5.get_shape()[1:]))], name="conv5_flatout")

            fc6 = nf.fc_layer(conv_5, model_params["fc6"], name="fc6", activat_fn=tf.nn.relu, reg=l2_reg)
            
            fc7 = nf.fc_layer(fc6, model_params["fc7"], name="fc7", activat_fn=tf.nn.relu, reg=l2_reg)
            
            dropout = tf.layers.dropout(fc7, rate=self.dropout, training=self.is_training, name='dropout2')
            
            fc_code = nf.fc_layer(dropout, model_params["fc_code"], name="fc_code", activat_fn=None, reg=l2_reg)
            
            return fc_code
        
    def alex_net_2D(self, kwargs = {}):

        #init = tf.random_normal_initializer(stddev=0.01)
        init = tf.contrib.layers.xavier_initializer()
        l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   
        
        model_params = {

                        'conv1': [11,11,96],
                        'conv2': [5,5,256],
                        'conv3': [3,3,384],
                        'conv4': [3,3,384],
                        'conv5': [3,3,256],        
                        
                        'conv6': [3,3,128],          
                        'conv7': [3,3,64],          
                        'conv_code': [3,3,16],                                  
                        #'conv_code': [3,3,1],                                  
                        }        
        with tf.name_scope("Detector"):  
            
            conv_1 = nf.convolution_layer(self.inputs, model_params["conv1"], [1,4,4,1], name="conv1", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')
            #conv_1 = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')
            conv_1 = tf.nn.max_pool(conv_1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
            
            conv_2 = nf.convolution_layer(conv_1, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')
            conv_2 = tf.nn.max_pool(conv_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

            conv_3 = nf.convolution_layer(conv_2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')

            conv_4 = nf.convolution_layer(conv_3, model_params["conv4"], [1,1,1,1], name="conv4", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')

            conv_5 = nf.convolution_layer(conv_4, model_params["conv5"], [1,1,1,1], name="conv5", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')
            
            conv_6 = nf.convolution_layer(conv_5, model_params["conv6"], [1,1,1,1], name="conv6", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')
            #conv_6 = tf.nn.max_pool(conv_6, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')            

            conv_7 = nf.convolution_layer(conv_6, model_params["conv7"], [1,1,1,1], name="conv7", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')
            #conv_7 = tf.nn.max_pool(conv_7, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')            

            dropout = tf.layers.dropout(conv_7, rate=self.dropout, training=self.is_training, name='dropout2')
            
            conv_code = nf.convolution_layer(dropout, model_params["conv_code"], [1,1,1,1], name="conv_code", activat_fn=tf.nn.relu, initializer=init, reg=l2_reg, padding='SAME')
            
            return conv_code        
            
    def build_model(self, kwargs = {}):

        model_list = ["googleLeNet_v1", "resNet_v1", "baseline", "alex_net", "alex_net_2D", "baseline_2D"]
        
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
