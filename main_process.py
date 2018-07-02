# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:18:05 2017

@author: Weiyu_Lee
"""

from autoencoder_model import AUTOENCODER_MODEL
from detection_model import DETECTION_MODEL

import tensorflow as tf
import argparse
import os
import config

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="example",help="Configuration name")
args = parser.parse_args()

conf = config.config(args.config).config["train"]

def main(_):

    if not os.path.exists(conf["checkpoint_dir"]):
        os.makedirs(conf["checkpoint_dir"])
    if not os.path.exists(conf["output_dir"]):
        os.makedirs(conf["output_dir"])

    with tf.Session() as sess:
        if conf["mode"] is "autoencoder":
        
            model = AUTOENCODER_MODEL(  sess, 
                                        mode=conf["mode"],
                                        is_train=conf["is_train"],                      
                                        iteration=conf["iteration"],
                                        curr_iteration=conf["curr_iteration"],
                                        batch_size=conf["batch_size"],
                                        image_size=conf["image_size"], 
                                        project_image_size=conf["project_image_size"], 
                                        learning_rate=conf["learning_rate"],
                                        checkpoint_dir=conf["checkpoint_dir"],
                                        ckpt_name=conf["ckpt_name"],
                                        log_dir=conf["log_dir"],
                                        output_dir=conf["output_dir"],
                                        model_ticket=conf["model_ticket"])
            
        elif conf["mode"] is "detection":

            model = DETECTION_MODEL(    sess, 
                                        mode=conf["mode"],
                                        is_train=conf["is_train"],                      
                                        iteration=conf["iteration"],
                                        curr_iteration=conf["curr_iteration"],
                                        batch_size=conf["batch_size"],
                                        image_size=conf["image_size"], 
                                        project_image_size=conf["project_image_size"], 
                                        learning_rate=conf["learning_rate"],
                                        checkpoint_dir=conf["checkpoint_dir"],
                                        ckpt_name=conf["ckpt_name"],
                                        log_dir=conf["log_dir"],
                                        output_dir=conf["output_dir"],
                                        model_ticket=conf["model_ticket"],
                                        datasetroot=conf["datasetroot"])
            
        if conf["is_train"]:
            model.train()
    
if __name__ == '__main__':
  tf.app.run()