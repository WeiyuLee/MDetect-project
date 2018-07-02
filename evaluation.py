import os
import sys
sys.path.append('./utility')

import numpy as np
import argparse
import tensorflow as tf
import scipy.misc
import pickle
import utils 
import AE_model_zoo
import config

import datetime

class encode_decode:

    def __init__(self, config):

        eval_conf = config["evaluation"]

        #Set evaluation configuration
        self.input_path = eval_conf["inputroot"]
        self.label_path = eval_conf["labelroot"]
        self.code_path = eval_conf["coderoot"]
        
        self.image_size = eval_conf["image_size"]
        self.project_image_size = eval_conf["project_image_size"]
        self.code_size = eval_conf["code_size"]
        
        self.model_ticket = eval_conf["model_ticket"]
        
        self.ckpt_file = eval_conf["ckpt_file"]
        
        self.enc_output_dir = eval_conf["enc_output_dir"]
        self.dec_output_dir = eval_conf["dec_output_dir"]
        
        self.input_data_list = os.listdir(self.input_path)
        self.label_data_list = os.listdir(self.label_path)
        self.code_data_list = os.listdir(self.code_path)
        
    def load_image_data(self):
        
        input_data = []
        label_data = []
        
        for f in self.input_data_list:
            curr_input = scipy.misc.imread(os.path.join(self.input_path, f))
            curr_label = scipy.misc.imread(os.path.join(self.label_path, f))
            
            curr_image_flipud = np.flipud(curr_input)
            curr_code_flipud = np.flipud(curr_label)
            
            curr_image_fliplr = np.fliplr(curr_input)
            curr_code_fliplr = np.fliplr(curr_label)
    
            curr_image_rot90 = np.rot90(curr_input)
            curr_code_rot90 = np.rot90(curr_label)
    
            curr_image_rot180 = np.rot90(curr_image_rot90)
            curr_code_rot180 = np.rot90(curr_code_rot90)                   
            
            input_data.append(curr_input)
            input_data.append(curr_image_flipud)
            input_data.append(curr_image_fliplr)
            input_data.append(curr_image_rot90)
            input_data.append(curr_image_rot180)
            
            label_data.append(curr_label)
            label_data.append(curr_code_flipud)
            label_data.append(curr_code_fliplr)
            label_data.append(curr_code_rot90)
            label_data.append(curr_code_rot180)
        
        return input_data, label_data

    def load_pickle_data(self):

        image_data = []
        code_data = []
        list_data = []

        for f in self.code_data_list:
            f = open(os.path.join(self.code_path, f), "rb")
            curr_image = pickle.load(f)
            curr_code = pickle.load(f)
            curr_list = pickle.load(f)
            f.close()
            
            image_data.append(curr_image)
            code_data.append(curr_code)
            list_data.append(curr_list)
        
        # code_data.shape = (?, 64, 64, 4)
        return image_data, code_data, list_data

    #Load Tensorflow Model from check points
    def load_encode_model(self):

        tf.reset_default_graph() 
                
        self.inputs = tf.placeholder(tf.float32, [None, self.project_image_size, self.project_image_size, 1])

        mz = AE_model_zoo.model_zoo(self.inputs, 1., False, self.model_ticket)
        predict_op = mz.build_model({"mode":"encoder", "image_size":self.project_image_size})
        
        if type(predict_op) is tuple:
            self.predict_op = predict_op[0]
        else:
            self.predict_op = predict_op

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self.ckpt_file)
        print("Session created and Model restored")

    #Load Tensorflow Model from check points
    def load_decode_model(self):

        tf.reset_default_graph() 
                
        self.inputs = tf.placeholder(tf.float32, [None, self.code_size[0], self.code_size[1], self.code_size[2]])

        mz = AE_model_zoo.model_zoo(self.inputs, 1., False, self.model_ticket)
        predict_op = mz.build_model({"mode":"decoder", "image_size":self.project_image_size, "code":self.inputs})
        
        if type(predict_op) is tuple:
            self.predict_op = predict_op[0]
        else:
            self.predict_op = predict_op

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self.ckpt_file)
        print("Session created and Model restored")

    def split_image(self, input, label, sub_size):
        
        height, width = label.shape
        output_image = []
        output_label = []
        output_check_list = np.zeros([(height//sub_size)*(width//sub_size)])
        patch_count = 0
        
        for h in range(0, height-sub_size, sub_size):
            for w in range(0, width-sub_size, sub_size):    
                output_image.append(input[h:h+sub_size, w:w+sub_size])
                output_label.append(label[h:h+sub_size, w:w+sub_size])

                if output_label[-1].sum() > 50: 
                    output_check_list[patch_count] = 1                        
        
                patch_count += 1
        
        #print("output_label shape: {}".format(np.array(output_label).shape))
        
        return output_image, output_label, output_check_list

    def prediction(self, inputs):
     
        #run model in model_ticket_list and return prediction       
        predicted = self.sess.run(self.predict_op, feed_dict = {self.inputs:inputs})
        return predicted
    
    def run_encode(self):

        #run evaluation for images in input folder and save as image
        input_data, label_data = self.load_image_data()
        self.load_encode_model()
        util = utils.utility(self.image_size, self.image_size, self.project_image_size)
        
        progress = 1

        for i_idx in range(len(input_data)):
            
            # Split label to 256x256 as input
            split_input, split_label, check_list = self.split_image(input_data[i_idx], label_data[i_idx], self.image_size)

            # Project the coordinates to OA
            batch_project = util.projection(split_label, len(split_label))
            
            # Encode
            pred = self.prediction(batch_project)
            
            # Save code per image as pickle
            if not os.path.exists(self.enc_output_dir):
                os.makedirs(self.enc_output_dir)          
                
            #f = open(self.enc_output_dir + os.path.splitext(self.label_data_list[i_idx])[0] + '.pkl', 'wb')
            f = open(self.enc_output_dir + '{}.pkl'.format(i_idx), 'wb')
            pickle.dump(split_input, f, True)
            pickle.dump(pred, f, True)
            pickle.dump(check_list, f, True)
            f.close()
            
            print("Process:{}/{}".format(progress, len(label_data)))
            progress += 1

    def run_decode(self):

        #run evaluation for images in input folder and save as image
        image_data, code_data, list_data = self.load_pickle_data()
        self.load_decode_model()
        util = utils.utility(self.image_size, self.image_size, self.project_image_size)
        
        progress = 1

        for i_idx in range(len(code_data)):
            
            print("File: {}".format(self.code_data_list[i_idx]))
            
            # Decode
            starttime = datetime.datetime.now()
            pred = self.prediction(code_data[i_idx])
            endtime = datetime.datetime.now()
            print("NN Decode time: {}".format((endtime - starttime).seconds))
            
            # Reverse to coordinates
            starttime = datetime.datetime.now()
            project_rev = util.projection_reverse(pred, len(pred))   
            endtime = datetime.datetime.now()
            print("Reverse time: {}".format((endtime - starttime).seconds))
            
            # Save results
            starttime = datetime.datetime.now()
            self.save_output(image_data[i_idx], self.code_data_list[i_idx], flag="_image_")
            self.save_output(project_rev, self.code_data_list[i_idx], flag="_decode_")
            endtime = datetime.datetime.now()
            print("save_output time: {}".format((endtime - starttime).seconds))
            
            print(list_data[i_idx])
            
            print("Process:{}/{}".format(progress, len(code_data)))
            progress += 1

    def save_output(self, decode_images, file_name, flag=[]):
        
        if not os.path.exists(self.dec_output_dir):
            os.makedirs(self.dec_output_dir)                

        for p_idx in range(len(decode_images)):            
            scipy.misc.imsave(self.dec_output_dir + os.path.splitext(file_name)[0] + flag + '_patch_{}.png'.format(p_idx), decode_images[p_idx].squeeze())  
            
def  main_process():

    #Parsing argumet(configuration name) from shell, 
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="example",help="Configuration name")
    args = parser.parse_args()
    conf = config.config(args.config).config
    
    enc_dec = encode_decode(conf)
    
    if conf["evaluation"]["mode"] is "encode":
        enc_dec.run_encode()
    elif conf["evaluation"]["mode"] is "decode":
        enc_dec.run_decode()

if __name__ == '__main__':

    main_process()
