import os
import sys
sys.path.append('./utility')

import numpy as np
import argparse
import tensorflow as tf
import scipy.misc
import pickle
import random
import datetime

import utils 
import AE_model_zoo
import DE_model_zoo
import config

class encode_decode_detect:

    def __init__(self, config):

        eval_conf = config["evaluation"]
        
        #Set evaluation configuration
        self.input_path = eval_conf["inputroot"]
        self.label_path = eval_conf["labelroot"]
        self.code_path = eval_conf["coderoot"]
        self.pred_path = eval_conf["predroot"]
        
        self.image_size = eval_conf["image_size"]
        self.project_image_size = eval_conf["project_image_size"]
        self.code_size = eval_conf["code_size"]
        
        self.model_ticket = eval_conf["model_ticket"]
        
        self.ckpt_file = eval_conf["ckpt_file"]
            
        self.input_data_list = os.listdir(self.input_path)
        self.label_data_list = os.listdir(self.label_path)
        self.code_data_list = os.listdir(self.code_path)

        self.model_list = ["baseline", "baseline_v2", "baseline_v3", "baseline_v4", "baseline_v5", "baseline_v5_flatten", "baseline_v6_flatten", 
                           "baseline_end2end", "baseline_end2end_2D", "baseline_end2end_2D_v2"]
        
        if eval_conf["mode"] is "encode" or eval_conf["mode"] is "decode": 
            self.sample_mode = eval_conf["sample_mode"]
            self.enc_output_dir = eval_conf["enc_output_dir"]
            self.dec_output_dir = eval_conf["dec_output_dir"]
            self.aug = eval_conf["aug"]
        
    def load_image_data(self):
        
        input_data = []
        label_data = []
        image_names = []
        
        for f in self.input_data_list:
            curr_input = scipy.misc.imread(os.path.join(self.input_path, f))
            curr_label = scipy.misc.imread(os.path.join(self.label_path, f))

            file_name = os.path.splitext(f)[0]

            input_data.append(curr_input)
            label_data.append(curr_label)            
            image_names.append(file_name)
            
            if self.aug is True:
                
#                curr_image_flipud = np.flipud(curr_input)
#                curr_code_flipud = np.flipud(curr_label)
                
#                curr_image_fliplr = np.fliplr(curr_input)
#                curr_code_fliplr = np.fliplr(curr_label)
        
                curr_image_rot90 = np.rot90(curr_input)
                curr_code_rot90 = np.rot90(curr_label)
        
                curr_image_rot180 = np.rot90(curr_image_rot90)
                curr_code_rot180 = np.rot90(curr_code_rot90)                   

                curr_image_rot270 = np.rot90(curr_image_rot180)
                curr_code_rot270 = np.rot90(curr_code_rot180)                   
                
#                input_data.append(curr_image_flipud)
#                image_names.append(file_name + "_flipud")                
#                input_data.append(curr_image_fliplr)
#                image_names.append(file_name + "_fliplr")                
#                input_data.append(curr_image_rot90)
                image_names.append(file_name + "_rot90")                
                image_names.append(file_name + "_rot180")                
                image_names.append(file_name + "_rot270")     

                input_data.append(curr_image_rot90)
                input_data.append(curr_image_rot180)                
                input_data.append(curr_image_rot270)
                
#                label_data.append(curr_code_flipud)
#                label_data.append(curr_code_fliplr)
                label_data.append(curr_code_rot90)
                label_data.append(curr_code_rot180)
                label_data.append(curr_code_rot270)                
        
        return input_data, label_data, image_names

    def load_pickle_data(self, input_path):

        image_data = []
        code_data = []
        list_data = []
        meta_data = []

        for f in self.code_data_list:
            
            if f == "mean_image.pkl":
                continue
            else:
                f = open(os.path.join(input_path, f), "rb")
                curr_image = pickle.load(f)
                curr_code = pickle.load(f)
                curr_list = pickle.load(f)
                curr_meata = pickle.load(f)            
                f.close()
                
                image_data.append(curr_image)
                code_data.append(curr_code)
                list_data.append(curr_list)
                meta_data.append(curr_meata)
        
        # code_data.shape = (?, 64, 64, 1) => (?, 4096)
        return image_data, code_data, list_data, meta_data

    def load_mean_image(self, data_root):
        
        f = open(os.path.join(data_root, "mean_image.pkl"), "rb")
        mean_image = pickle.load(f)
        f.close()
        
        return mean_image


    #Load Tensorflow Model from check points
    def load_encode_model(self):

        tf.reset_default_graph() 

        if self.model_ticket is "baseline_end2end" or self.model_ticket is "baseline_end2end_2D" or self.model_ticket is "baseline_end2end_2D_v2":                
            self.inputs = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1])
        else:
            self.inputs = tf.placeholder(tf.float32, [None, self.project_image_size[0], self.project_image_size[1], 1])
            
        mz = AE_model_zoo.model_zoo(self.inputs, 1., False, self.model_ticket)
        predict_op = mz.build_model({"model_list":self.model_list, "mode":"encoder", "image_size":[self.project_image_size[0], self.project_image_size[1]]})
        
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
        
        if self.model_ticket is "baseline_end2end_2D":        
            self.inputs = tf.placeholder(tf.float32, [None, self.code_size[0], self.code_size[1], self.code_size[2]])
        else:
            self.inputs = tf.placeholder(tf.float32, [None, self.code_size[0]*self.code_size[1]*self.code_size[2]])

        mz = AE_model_zoo.model_zoo(self.inputs, 1., False, self.model_ticket)
        
        if self.model_ticket is "baseline_end2end_2D":    
            predict_op = mz.build_model({"model_list":self.model_list, "mode":"decoder", "image_size":[self.image_size, self.image_size], "code":self.inputs})
        else:
            predict_op = mz.build_model({"model_list":self.model_list, "mode":"decoder", "image_size":[self.project_image_size[0], self.project_image_size[1]], "code":self.inputs})    
            
        if type(predict_op) is tuple:
            self.predict_op = predict_op[0]
        else:
            self.predict_op = predict_op

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self.ckpt_file)
        print("Session created and Model restored")
        
    #Load Tensorflow Model from check points
    def load_detection_model(self):

        tf.reset_default_graph() 
                
        self.inputs = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='input')

        mz = DE_model_zoo.model_zoo(self.inputs, 1., False, self.model_ticket)
        predict_op = mz.build_model()
        
        if type(predict_op) is tuple:
            self.predict_op = predict_op[0]
        else:
            self.predict_op = predict_op

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self.ckpt_file)
        print("Session created and Model restored")        

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
        
        output_image = np.array(output_image)
        mean_image = np.sum(output_image, axis=0) / output_image.shape[0]
        
        print("output_label shape: {}, residual parts: [h,w] = [{}, {}]".format(np.array(output_label).shape, residual_h_count, residual_w_count))
        print("output_meta_data shape: {}".format(len(output_meta_data)))
        
        return output_image, output_label, output_check_list, output_meta_data, mean_image

    def random_crop_image(self, input, label, image_name, sub_size, crop_num):
        
        height, width = label.shape
        output_image = []
        output_label = []
        output_check_list = np.zeros([crop_num])
        output_meta_data = []
        
        count = 0
        n_count = 0
        ab_count = 0
        
#        n_num = crop_num // 4
#        ab_num = n_num * 3
        n_num = crop_num//2
        ab_num = crop_num//2
        
        while (count < crop_num):
            h = random.randint(0, height-sub_size)
            w = random.randint(0, width-sub_size)

            temp_label = label[h:h+sub_size, w:w+sub_size]   
    
            # abnormal            
            if temp_label.sum() > 0: 
                if ab_count < ab_num:
                    output_label.append(temp_label)        
                    output_image.append(input[h:h+sub_size, w:w+sub_size])
                    output_meta_data.append((image_name, [h,w], "random_grid", "abnormal"))
                    
                    output_check_list[count] = 1                        
                    ab_count += 1
                    count += 1
                else:
                    continue
                
            # normal
            else:
                if n_count < n_num:
                    output_label.append(temp_label)        
                    output_image.append(input[h:h+sub_size, w:w+sub_size])
                    output_meta_data.append((image_name, [h,w], "random_grid", "normal"))
                    
                    n_count += 1
                    count += 1
                else:
                    continue;                                     

        output_image = np.array(output_image)

        print("normal: {}, abnormal: {}".format(n_count, ab_count))
                    
        print("output_label shape: {}".format(np.array(output_label).shape))
        
        return output_image, output_label, output_check_list, output_meta_data

    def data_stat(self, input_code, table):
        # record the cell number in each patch
        
        print("Data shape: [{}]".format(input_code.shape))
        
        for i_idx in range(len(input_code)):
            curr_code_sum = input_code[i_idx].sum()
            try:
                table[curr_code_sum] += 1
            except:
                table[curr_code_sum] = 1
                
        return table

    def prediction(self, inputs):
     
        #run model in model_ticket_list and return prediction       
        predicted = self.sess.run(self.predict_op, feed_dict = {self.inputs:inputs})
        return predicted
    
    def run_encode(self):

        #run evaluation for images in input folder and save as image
        input_data, label_data, image_names = self.load_image_data()
        self.load_encode_model()
        util = utils.utility(self.image_size, self.image_size, self.project_image_size[0])
        
        batch_size = 32
        progress = 1

        minimum = 255
        maximum = 0

        ab_table = {}
        
        mean_image = np.zeros((self.image_size, self.image_size, 3))

        for i_idx in range(len(input_data)):

            pred = []
            
            # Split label to 256x256 as input
            if self.sample_mode is "split":
                split_input, split_label, check_list, meta_data, curr_mean_image = self.split_image(input_data[i_idx], label_data[i_idx], image_names[i_idx], self.image_size)
            elif self.sample_mode is "random":
                split_input, split_label, check_list, meta_data = self.random_crop_image(input_data[i_idx], label_data[i_idx], image_names[i_idx], self.image_size, 512)

            mean_image += curr_mean_image

            if self.model_ticket is "baseline_end2end" or self.model_ticket is "baseline_end2end_2D" or self.model_ticket is "baseline_end2end_2D_v2":
                batch_project = np.array(split_label)
                batch_project = np.expand_dims(batch_project, axis=-1)
                
                ab_table = self.data_stat(batch_project, ab_table)
            else:
                # Project the coordinates to OA
                batch_project = util.projection(split_label, len(split_label))

            sample_num = batch_project.shape[0]
            
            # Encode
            for b_idx in range(0, sample_num-batch_size+1, batch_size):
                curr_batch = batch_project[b_idx:b_idx+batch_size]
                curr_pred = self.prediction(curr_batch)
                
                if  maximum < np.array(curr_pred).max():
                    maximum = np.array(curr_pred).max()
                if  minimum > np.array(curr_pred).min():
                    minimum = np.array(curr_pred).min()                
                    
                pred.extend(curr_pred)
            
            if (sample_num % batch_size) != 0:
                curr_batch = batch_project[-(sample_num % batch_size):]
                curr_pred = self.prediction(curr_batch)
                pred.extend(curr_pred)                              
            
            print(np.shape(pred))
                        
            # Save code per image as pickle
            if not os.path.exists(self.enc_output_dir):
                os.makedirs(self.enc_output_dir)          
                
            #f = open(self.enc_output_dir + os.path.splitext(self.label_data_list[i_idx])[0] + '.pkl', 'wb')
            f = open(self.enc_output_dir + '{}.pkl'.format(image_names[i_idx]), 'wb')
            pickle.dump(split_input, f, True)
            pickle.dump(pred, f, True)
            pickle.dump(check_list, f, True)
            pickle.dump(meta_data, f, True)
            f.close()
            
            print("Process:{}/{}".format(progress, len(label_data)))
            progress += 1
        
        mean_image = mean_image / len(input_data)
        f = open(self.enc_output_dir + 'mean_image.pkl', 'wb')
        pickle.dump(mean_image, f, True)
        f.close()
            
        print("[Min:Max] = [{}, {}]".format(minimum, maximum))            
        print(ab_table)

    def run_decode(self):

        #run evaluation for images in input folder and save as image
        image_data, code_data, list_data, meta_data = self.load_pickle_data(self.pred_path)
        self.load_decode_model()
        util = utils.utility(self.image_size, self.image_size, self.project_image_size[0])
        
        progress = 1

        for i_idx in range(len(code_data)):
            
            if self.code_data_list[i_idx] == "mean_image.pkl":
                self.code_data_list = np.delete(self.code_data_list, i_idx)
            
            print("========== File: {} ========== ".format(self.code_data_list[i_idx]))
            
            # Decode
            starttime = datetime.datetime.now()
            pred = self.prediction(code_data[i_idx])
            endtime = datetime.datetime.now()
            print("NN Decode time: {}".format((endtime - starttime).seconds))
            
            if self.model_ticket is "baseline_end2end_2D":
                project_rev = pred
            else:                    
                # Reverse to coordinates
                starttime = datetime.datetime.now()
                project_rev = util.projection_reverse(pred, len(pred), debug_msg=24)   
                endtime = datetime.datetime.now()
                print("Reverse time: {}".format((endtime - starttime).seconds))
            
            # Save results
            starttime = datetime.datetime.now()
            
            file_name = os.path.splitext(self.code_data_list[i_idx])[0]
            self.save_output(project_rev, 
                             os.path.join(self.dec_output_dir, file_name, 'mask'), 
                             file_name, 
                             norm=False)
            
            self.save_output(project_rev, 
                             os.path.join(self.dec_output_dir, file_name, 'mask_norm'), 
                             file_name, 
                             norm=True)
            
            self.save_output(image_data[i_idx],
                             os.path.join(self.dec_output_dir, file_name, 'image'), 
                             file_name, 
                             norm=True)
            
            endtime = datetime.datetime.now()
            print("save_output time: {}".format((endtime - starttime).seconds))
            
            ab_idx = np.nonzero(list_data[i_idx])[0]
            print(ab_idx)
            
            print("Process:{}/{}".format(progress, len(code_data)))
            progress += 1

    def run_detection(self):

        #run detection for images in input folder and save as image
        image_data, code_data, list_data, meta_data = self.load_pickle_data(self.code_path)
        image_data = np.array(image_data)
        code_data = np.array(code_data)
        
        self.load_detection_model()
        
        # Normalization 
        mean_image = self.load_mean_image("/data/wei/dataset/MDetection/ICPR2012/training_data/scanner_A/label_code/256x256/")
        image_data = image_data - mean_image
        min_value = np.min(image_data)
        image_data = image_data - min_value
        max_value = np.max(image_data)
        image_data = image_data / max_value * 255

        
        progress = 1

        for i_idx in range(len(image_data)):
            
            #print("File: {}".format(self.code_data_list[i_idx]))
            
            # Detection
            pred = self.prediction(image_data[i_idx])

            # Reverse
            pred = (pred * config.mean_std[self.image_size][3]) + config.mean_std[self.image_size][2]

            # Save code per image as pickle
            if not os.path.exists(self.pred_path):
                os.makedirs(self.pred_path) 
                
            f = open(self.pred_path + '{}.pkl'.format(meta_data[i_idx][0][0]), 'wb')
            pickle.dump(image_data[i_idx], f, True)
            pickle.dump(pred, f, True)
            pickle.dump(list_data[i_idx], f, True)
            pickle.dump(meta_data[i_idx], f, True)
            f.close()
            
            print("Process:{}/{}".format(progress, len(code_data)))
            progress += 1

    def save_output(self, decode_images, dir_path, file_name, norm=False):
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)                

        for p_idx in range(len(decode_images)):   
            
            curr_image = decode_images[p_idx].squeeze()
            
            if norm == True:
                scipy.misc.imsave(os.path.join(dir_path, file_name + '_patch_{}.png'.format(p_idx)), curr_image)  
            else:
                scipy.misc.toimage(curr_image, cmin=0, cmax=255).save(os.path.join(dir_path, file_name + '_patch_{}.png'.format(p_idx)))
            
def  main_process():

    #Parsing argumet(configuration name) from shell, 
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="example",help="Configuration name")
    args = parser.parse_args()
    conf = config.config(args.config).config
    
    enc_dec_detect = encode_decode_detect(conf)
    
    if conf["evaluation"]["mode"] is "encode":
        print("=================================")
        print("=            Encoder            =")
        print("=================================")        
        enc_dec_detect.run_encode()
    elif conf["evaluation"]["mode"] is "decode":
        print("=================================")
        print("=            Decoder            =")
        print("=================================")
        enc_dec_detect.run_decode()
    elif conf["evaluation"]["mode"] is "detection":
        print("=================================")
        print("=           Detection           =")
        print("=================================")
        enc_dec_detect.run_detection()   

if __name__ == '__main__':

    main_process()
