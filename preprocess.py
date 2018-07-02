# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:19:44 2018

@author: Weiyu_Lee
"""

import os
import pickle
import numpy as np
       
def load_pickle_data(datasetroot):

    image_data = []
    code_data = []
    check_list_data = []
    
    data_list = os.listdir(datasetroot)
    
    for f in data_list:
        f = open(os.path.join(datasetroot, f), "rb")
        curr_image = pickle.load(f)
        curr_code = pickle.load(f)
        curr_list = pickle.load(f)
        f.close()
        
        image_data.append(curr_image)
        code_data.append(curr_code)
        check_list_data.append(curr_list)
    
    # code_data.shape = (?, 64, 64, 4)
    return image_data, code_data, check_list_data   

def classify_data(image_data, code_data, check_list_data, datasetroot):
    
    normal_image_data = []
    normal_code_data = []
    abnormal_image_data = []
    abnormal_code_data = []

    ab_count = 0
    total_ab_count = 0
    
    data_list = os.listdir(datasetroot)
    
    for i_idx in range(len(data_list)):
               
        curr_list = check_list_data[i_idx]
        
        ab_idx = np.nonzero(curr_list)[0]

        for p_idx in range(len(curr_list)):
            if p_idx in ab_idx:
                abnormal_image_data.append(image_data[i_idx][p_idx])
                abnormal_code_data.append(code_data[i_idx][p_idx])
                ab_count += 1
            else:
                normal_image_data.append(image_data[i_idx][p_idx])
                normal_code_data.append(code_data[i_idx][p_idx])             

        print("File: {}, abnormal patch count: {}".format(data_list[i_idx], ab_count))        
        print(ab_idx)
        total_ab_count += ab_count
        ab_count = 0

    print("Total abnormal patch count: {}".format(total_ab_count))
                                
    return normal_image_data, normal_code_data, abnormal_image_data, abnormal_code_data

def aug_data(image_data, code_data):
    
    aug_image_data = []
    aug_code_data = []

    for i_idx in range(len(image_data)):
        curr_image = image_data[i_idx]
        curr_code = code_data[i_idx]
        
        curr_image_flipud = np.flipud(curr_image)
        curr_code_flipud = np.flipud(curr_code)
        
        curr_image_fliplr = np.fliplr(curr_image)
        curr_code_fliplr = np.fliplr(curr_code)

        curr_image_rot90 = np.rot90(curr_image)
        curr_code_rot90 = np.rot90(curr_code)

        curr_image_rot180 = np.rot90(curr_image_rot90)
        curr_code_rot180 = np.rot90(curr_code_rot90)
        
        aug_image_data.append(curr_image_flipud)
        aug_image_data.append(curr_image_fliplr)
        aug_image_data.append(curr_image_rot90)
        aug_image_data.append(curr_image_rot180)

        aug_code_data.append(curr_code_flipud)
        aug_code_data.append(curr_code_fliplr)
        aug_code_data.append(curr_code_rot90)
        aug_code_data.append(curr_code_rot180)
        
    image_data.extend(aug_image_data)
    code_data.extend(aug_code_data)
    
    return image_data, code_data
        
def main_process():
    
    pickle_import_dir = "/home/sdc1/dataset/ICPR2012/training_data/scanner_A/label_code/"    
    output_dir = "./classfied_data/"
    
    print("Loading data...")
    image_data, code_data, check_list_data = load_pickle_data(pickle_import_dir)
    
    print("Classifying data...")
    n_image, n_code, ab_image, ab_code = classify_data(image_data, code_data, check_list_data, pickle_import_dir)
    
#    print("Data augmentation...")
#    ab_image, ab_code = aug_data(ab_image, ab_code)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)       
    
    print("Normal data: {}".format(len(n_image)))
    print("Abnormal data: {}".format(len(ab_image)))
    
    print("Dumping data...")
    f = open(output_dir + "normal_data.pkl", 'wb')
    pickle.dump(n_image, f, True)
    pickle.dump(n_code, f, True)
    f.close()

    f = open(output_dir + "abnormal_data.pkl", 'wb')
    pickle.dump(ab_image, f, True)
    pickle.dump(ab_code, f, True)
    f.close()
    
if __name__ == '__main__':

    main_process()    