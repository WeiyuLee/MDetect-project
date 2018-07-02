# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:03:19 2017

@author: Weiyu_Lee
"""

import math
import numpy as np
import random
import tensorflow as tf

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

class utility:

    def __init__(self, image_width, image_height, project_size):
        
        self.image_width = image_width
        self.image_height = image_height
        self.num_lines = project_size
        #self.R = round(math.sqrt((self.image_width/2)**2 + (self.image_height/2)**2))
        [self.R, self.X, self.Y, self.A, self.B, self.Z, self.radian] = self.build_LTable()        
        
    def build_LTable(self):

        radian = []
        X = []
        Y = []
        A = []
        B = []
        Z = []
        
        R = round(math.sqrt((self.image_width/2)**2 + (self.image_height/2)**2))
        #num_lines = 2*R+1
        
        for line_idx in range(self.num_lines):
            radian.append((2*math.pi/self.num_lines) * line_idx)
            
            A.append(math.cos(radian[-1]))
            B.append(math.sin(radian[-1]))
            
            X.append(self.image_width/2  - R*B[-1])
            Y.append(self.image_height/2 + R*A[-1])    
            
            Z = (A[-1]**2 + B[-1]**2)
            
        return R, X, Y, A, B, Z, radian
    
    def get_batch_data(self, batch_size):
        
        max_cell_num = 5
        output_batch_image = []
        output_batch_project = []
    
        #R = round(math.sqrt((self.image_width/2)**2 + (self.image_height/2)**2))
        #num_lines = 2*self.R+1
        #radian = []
        #for line_idx in range(num_lines):
        #    radian.append((2*math.pi/num_lines) * line_idx)
        
        for b_idx in range(batch_size):
            curr_cell_num = random.randint(0, max_cell_num)
            #curr_cell_num = max_cell_num
            
            curr_image = np.zeros([self.image_height, self.image_width, 1])
            # Enlarge the projection size to 384x384, due to the detection model's output matching
            # The original size should be [self.num_lines, 2*self.R+1, 1]            
            curr_project_image = np.zeros([self.num_lines, self.num_lines, 1])
            for c_idx in range(curr_cell_num):
                h_coordinate = random.randint(0, self.image_height-1)
                w_coordinate = random.randint(0, self.image_width-1)
                
                curr_image[h_coordinate, w_coordinate] = 255
                
                # Projection
                for line_idx in range(self.num_lines):
                    #X = (self.image_width/2  - R*math.sin(radian[line_idx]))
                    #Y = (self.image_height/2 + R*math.cos(radian[line_idx]))
                    
#                    r = round( (h_coordinate - self.X[line_idx]) * math.cos(self.radian[line_idx]) + 
#                               (w_coordinate - self.Y[line_idx]) * math.sin(self.radian[line_idx])   )
#                    d = -(h_coordinate - X) * math.sin(radian[line_idx]) + (w_coordinate - Y) * math.cos(radian[line_idx])

                    r = round( (h_coordinate - self.X[line_idx]) * self.A[line_idx] + (w_coordinate - self.Y[line_idx]) * self.B[line_idx] )
                    d = -(-(h_coordinate - self.X[line_idx]) * self.B[line_idx] + (w_coordinate - self.Y[line_idx]) * self.A[line_idx])
                    
#                    Z = (self.A[line_idx]**2 + self.B[line_idx]**2)
#                    re_h = round( self.X[line_idx] + self.A[line_idx]*r/Z + self.B[line_idx]*d/Z )
#                    re_w = round( self.Y[line_idx] + self.B[line_idx]*r/Z - self.A[line_idx]*d/Z )
                    
#                    print("[{}, {}], [{}, {}]".format(h_coordinate, w_coordinate, re_h, re_w))
                    
                    # Use (self.num_lines//2) as the center point instead of self.R
                    #curr_project_image[line_idx, r+self.R] = d                                                        
                    curr_project_image[line_idx, r+(self.num_lines//2)] = d                                                        
            
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
            curr_project_image = np.zeros([self.num_lines, self.num_lines, 1]) 
            for c_idx in range(len(coordinates[0])):
                h_coordinate = coordinates[0][c_idx]
                w_coordinate = coordinates[1][c_idx]

                for line_idx in range(self.num_lines):
                    r = round( (h_coordinate - self.X[line_idx]) * self.A[line_idx] + (w_coordinate - self.Y[line_idx]) * self.B[line_idx] )
                    d = -(-(h_coordinate - self.X[line_idx]) * self.B[line_idx] + (w_coordinate - self.Y[line_idx]) * self.A[line_idx])            

                    # Use (self.num_lines//2) as the center point instead of self.R    
                    curr_project_image[line_idx, r+(self.num_lines//2)] = d 
                    
            output_batch_project.append(curr_project_image)    
        
        #print("output_batch_project shape: {}".format(np.array(output_batch_project).shape))
        
        return np.array(output_batch_project)                
            
    
    def projection_reverse(self, batch_project_image, batch_size):
        
        num_lines = batch_project_image[0].shape[0]
        
        output_batch_image = []
        
        for b_idx in range(batch_size):
            curr_project_image = batch_project_image[b_idx]
            curr_image = np.zeros([self.image_height, self.image_width, 1])
            
            for line_idx in range(num_lines):
                curr_row = curr_project_image[line_idx, :]
                
                # record the nonzero idx
                cells = np.nonzero(curr_row)
                
                for cell_idx in cells[0]:
                    #r = cell_idx - self.R
                    r = cell_idx - (self.num_lines//2)
                    d = curr_row[cell_idx, 0]        
                    
                    #Z = (self.A[line_idx]**2 + self.B[line_idx]**2)
                    re_h = round( self.X[line_idx] + self.A[line_idx]*r/self.Z + self.B[line_idx]*d/self.Z )
                    re_w = round( self.Y[line_idx] + self.B[line_idx]*r/self.Z - self.A[line_idx]*d/self.Z )

                    if (re_h >= 0 and re_h < self.image_height) and (re_w >= 0 and re_w < self.image_width):
                        curr_image[re_h, re_w] = 255
            
            output_batch_image.append(curr_image)
            
        return np.array(output_batch_image)

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator







