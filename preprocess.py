# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:19:44 2018

@author: Weiyu_Lee
"""

import os
import pickle
import numpy as np

#    256x256
#    Image [mean, std]: [157.15577773758343, 57.6469560215388]
#    Code  [mean, std]: [9.645856857299805, 9.694955825805664]   
    
#    224x224
#    Image [mean, std]: [157.0940914357948, 57.67350522847466]
#    Code  [mean, std]: [9.611194610595703, 8.928753852844238]
       
def load_pickle_data(datasetroot, data_list):

    image_data = []
    code_data = []
    check_list_data = []
    meta_data = []
    
#    data_list = os.listdir(datasetroot)
    
    for f in data_list:
        f = open(os.path.join(datasetroot, f), "rb")
        curr_image = pickle.load(f)
        curr_code = pickle.load(f)
        curr_list = pickle.load(f)
        curr_meta = pickle.load(f)
        f.close()
        
        image_data.append(curr_image)
        code_data.append(curr_code)
        check_list_data.append(curr_list)
        meta_data.append(curr_meta)
    
    # code_data.shape = (?, 64, 64, 4)
    return image_data, code_data, check_list_data, meta_data   

def classify_data(image_data, code_data, check_list_data, meta_data, data_list):
    
    normal_image_data = []
    normal_code_data = []
    normal_meta_data = []
    abnormal_image_data = []
    abnormal_code_data = []
    abnormal_meta_data = []
    
    ab_count = 0
    total_ab_count = 0
    
#    data_list = os.listdir(datasetroot)
    
    for i_idx in range(len(data_list)):
               
        curr_list = check_list_data[i_idx]
        
        ab_idx = np.nonzero(curr_list)[0]

        #print(curr_list)
        #print(np.array(image_data).shape)
        #print(np.array(code_data).shape)
        #print(len(meta_data))

        for p_idx in range(len(curr_list)):
            if p_idx in ab_idx:
                abnormal_image_data.append(image_data[i_idx][p_idx])
                abnormal_code_data.append(code_data[i_idx][p_idx])
                abnormal_meta_data.append(meta_data[i_idx][p_idx])
                ab_count += 1
            else:
                normal_image_data.append(image_data[i_idx][p_idx])
                normal_code_data.append(code_data[i_idx][p_idx])             
                normal_meta_data.append(meta_data[i_idx][p_idx])
                
        print("File: {}, abnormal patch count: {}".format(data_list[i_idx], ab_count))        
        print(ab_idx)
        total_ab_count += ab_count
        ab_count = 0

    print("Total abnormal patch count: {}".format(total_ab_count))
                                
    return normal_image_data, normal_code_data, normal_meta_data, abnormal_image_data, abnormal_code_data, abnormal_meta_data

def normalize_data(image_data, code_data, mean_std):
   
    image_data = np.array(image_data)
    code_data = np.array(code_data)
    
#    image_mean = np.mean(image_data)
#    image_std = np.std(image_data)
#    
#    code_mean = np.mean(code_data)
#    code_std = np.std(code_data)

    image_mean = mean_std[0]
    image_std = mean_std[1]
    code_mean = mean_std[2]
    code_std = mean_std[3]
   
    image_data = image_data - image_mean
    image_data = image_data / image_std
    code_data = (code_data - code_mean) / code_std

    print("Image [mean, std]: [{}, {}]".format(image_mean, image_std))
    print("Code  [mean, std]: [{}, {}]".format(code_mean, code_std))
       
    return image_data, code_data
        
def main_process():
    
    pickle_import_dir = "/data/wei/dataset/MDetection/ICPR2012/training_data/scanner_A/label_code/224x224"    
    output_dir = "/data/wei/dataset/MDetection/ICPR2012/training_data/scanner_A/classfied_data/"
#    pickle_import_dir = "/home/sdc1/dataset/ICPR2012/testing_data/scanner_A/label_code/"    
#    output_dir = "/home/sdc1/dataset/ICPR2012/testing_data/scanner_A/classfied_data/"

#    256x256 
#    image_mean = 157.15577773758343
#    image_std = 57.6469560215388
#    code_mean = 9.645856857299805
#    code_std = 9.694955825805664

#    224x224 
    image_mean = 157.0940914357948
    image_std = 57.67350522847466
    code_mean = 9.611194610595703
    code_std = 8.928753852844238

    mean_std = [image_mean, image_std, code_mean, code_std]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    
    print("Loading data...")
    print(pickle_import_dir)    
    data_list = os.listdir(pickle_import_dir)
    data_num = len(data_list)
    data_step = data_num//5
    
    batch_count = 0
    for d_idx in range(0, data_num, data_step):
        
        print("============= Batch: [{}] {}:{} =============".format(batch_count, d_idx, d_idx+data_step))
        
        curr_data_list = data_list[d_idx:d_idx+data_step]
        
        image_data, code_data, check_list_data, meta_data = load_pickle_data(pickle_import_dir, curr_data_list)
    
#        print("Normalizing data...{}".format(batch_count))
#        image_data, code_data = normalize_data(image_data, code_data, mean_std)
        
        print("Classifying data...{}".format(batch_count))
        n_image, n_code, n_meta, ab_image, ab_code, ab_meta = classify_data(image_data, code_data, check_list_data, meta_data, curr_data_list) 
        
        print("Normal data: {}".format(len(n_image)))
        print("Abnormal data: {}".format(len(ab_image)))
        
        print("Dumping data...{}".format(batch_count))
        f = open(output_dir + "normal_data_{}.npy".format(batch_count), 'wb')
        np.save(f, n_image)
        np.save(f, n_code)
        f.close()   
        f = open(output_dir + "normal_meta_data_{}.pkl".format(batch_count), 'wb')
        pickle.dump(n_meta, f, True)
        f.close() 
        
        f = open(output_dir + "abnormal_data_{}.npy".format(batch_count), 'wb')
        np.save(f, ab_image)
        np.save(f, ab_code)
        f.close()
        f = open(output_dir + "abnormal_meta_data_{}.pkl".format(batch_count), 'wb')
        pickle.dump(ab_meta, f, True)
        f.close()
    
        batch_count += 1
    
if __name__ == '__main__':

    main_process()    
    
    
    
    
    
    
    
    
    