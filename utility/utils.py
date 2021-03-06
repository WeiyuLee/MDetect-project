# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:03:19 2017

@author: Weiyu_Lee
"""

import math
import numpy as np
import random
import tensorflow as tf
import datetime

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

class utility:

    def __init__(self, image_width, image_height, project_size):
        
        self.image_width = image_width
        self.image_height = image_height
        self.num_lines = project_size
        [self.R, self.X, self.Y, self.A, self.B, self.Z, self.radian] = self.build_LTable()        
        
    def build_LTable(self):

        radian = []
        X = []
        Y = []
        A = []
        B = []
        Z = []
        
        R = round(math.sqrt((self.image_width/2)**2 + (self.image_height/2)**2))
        
        for line_idx in range(self.num_lines):
            radian.append((2*math.pi/self.num_lines) * line_idx)
            
            A.append(math.cos(radian[-1]))
            B.append(math.sin(radian[-1]))
            
            X.append(self.image_width/2  - R*B[-1])
            Y.append(self.image_height/2 + R*A[-1])    
            
            Z = (A[-1]**2 + B[-1]**2)
            
        return R, X, Y, A, B, Z, radian
    
    def get_batch_data(self, batch_size, project_enable=True):
        
        max_cell_num = 4
        output_batch_image = []
        output_batch_project = []
        
        for b_idx in range(batch_size):
            curr_cell_num = random.randint(0, max_cell_num)
            #curr_cell_num = max_cell_num
            
            curr_image = np.zeros([self.image_height, self.image_width, 1])
            # Enlarge the projection size to 384x384, due to the detection model's output matching
            # The original size should be [self.num_lines, 2*self.R+1, 1]            
            curr_project_image = np.zeros([self.num_lines, 2*self.R+1, 1])
            #curr_project_image = np.zeros([self.num_lines, self.num_lines, 1])
            for c_idx in range(curr_cell_num):
                h_coordinate = random.randint(0, self.image_height-1)
                w_coordinate = random.randint(0, self.image_width-1)
                
                curr_image[h_coordinate, w_coordinate] = 255
                
                if project_enable is True:
                    # Projection
                    for line_idx in range(self.num_lines):
    
                        r = round( (h_coordinate - self.X[line_idx]) * self.A[line_idx] + (w_coordinate - self.Y[line_idx]) * self.B[line_idx] )
                        d = -(-(h_coordinate - self.X[line_idx]) * self.B[line_idx] + (w_coordinate - self.Y[line_idx]) * self.A[line_idx])
                        
                        # Use (self.num_lines//2) as the center point instead of self.R
                        curr_project_image[line_idx, r+self.R] = d                                                        
                        #curr_project_image[line_idx, r+(self.num_lines//2)] = d                                                        
            
            output_batch_image.append(curr_image)
            output_batch_project.append(curr_project_image)
        
        #print("Max: {}".format(np.array(output_batch_project).max()))
        #print("Min: {}".format(np.array(output_batch_project).min()))
        
        return np.array(output_batch_image), np.array(output_batch_project)  
    
    def projection(self, batch_image, batch_size):
        
        output_batch_project = []
        
        for b_idx in range(batch_size):
            curr_image = batch_image[b_idx]
            coordinates = curr_image.nonzero()
            
            # Enlarge the projection size to 384x384, due to the detection model's output matching
            # The original size should be [self.num_lines, 2*self.R+1, 1]
            curr_project_image = np.zeros([self.num_lines, 2*self.R+1, 1]) 
            #curr_project_image = np.zeros([self.num_lines, self.num_lines, 1]) 
            for c_idx in range(len(coordinates[0])):
                h_coordinate = coordinates[0][c_idx]
                w_coordinate = coordinates[1][c_idx]

                for line_idx in range(self.num_lines):
                    r = round( (h_coordinate - self.X[line_idx]) * self.A[line_idx] + (w_coordinate - self.Y[line_idx]) * self.B[line_idx] )
                    d = -(-(h_coordinate - self.X[line_idx]) * self.B[line_idx] + (w_coordinate - self.Y[line_idx]) * self.A[line_idx])            

                    # Use (self.num_lines//2) as the center point instead of self.R    
                    curr_project_image[line_idx, r+self.R] = d        
                    #curr_project_image[line_idx, r+(self.num_lines//2)] = d 
                    
            output_batch_project.append(curr_project_image)    
            
        output_batch_project = np.array(output_batch_project)               
        
        #print("output_batch_project shape: {}".format(np.array(output_batch_project).shape))
        
        return output_batch_project
            
    
    def projection_reverse(self, batch_project_image, batch_size, debug_msg=0):
        
        num_lines = batch_project_image[0].shape[0]
        
        output_batch_image = []
        
        #print("num_lines = [{}]".format(num_lines))
        #print("batch_size = [{}]".format(batch_size))
        
        for b_idx in range(batch_size):
            curr_project_image = batch_project_image[b_idx]
            curr_image = np.zeros([self.image_height, self.image_width, 1])

            for line_idx in range(num_lines):
                curr_row = curr_project_image[line_idx, :]
                
                # record the nonzero idx
                #cells = np.nonzero(curr_row)                       
                cells = np.where(np.abs(curr_row) > 0.9)                       
                
                for cell_idx in cells[0]:
                    r = cell_idx - self.R
                    #r = cell_idx - (self.num_lines//2)
                    d = curr_row[cell_idx, 0]                        
                    
                    #Z = (self.A[line_idx]**2 + self.B[line_idx]**2)
                    re_h = round( self.X[line_idx] + ((self.A[line_idx]*r + self.B[line_idx]*d) / self.Z) )
                    re_w = round( self.Y[line_idx] + ((self.B[line_idx]*r - self.A[line_idx]*d) / self.Z) )            
                    
                    if (re_h >= 0 and re_h < self.image_height) and (re_w >= 0 and re_w < self.image_width):
                        curr_image[re_h, re_w] += 1
                
            output_batch_image.append(curr_image)
        
        output_batch_image = np.array(output_batch_image)
        
        return output_batch_image

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator







