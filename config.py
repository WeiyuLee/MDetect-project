import math

# 256x256 
image_mean_256 = 157.15577773758343
image_std_256 = 57.6469560215388
code_mean_256 = 9.645856857299805
code_std_256 = 9.694955825805664

# 224x224 
image_mean_224 = 157.0940914357948
image_std_224 = 57.67350522847466
code_mean_224 = 9.611194610595703
code_std_224 = 8.928753852844238

mean_std = { 256: [image_mean_256, image_std_256, code_mean_256, code_std_256],
             224: [image_mean_256, image_std_256, code_mean_256, code_std_256] }

file_root = '/data/wei/'
#file_root = '/home/sdc1/'

#data_type = 'training_data'
data_type = 'testing_data'

class config:

	def __init__(self, configuration):
		
		self.configuration = configuration
		self.config = {
						"common":{},
						"preprocess":{},
						"train":{},
						"evaluation":{
								"dataroot":None,
								"test_set":["Set5", "Set14", "BSD100"],
								"models":{},
								
							}
						}
		self.get_config()
		print("File root: {}".format(file_root))    
		print("Data type: {}".format(data_type))    

	def get_config(self):

		try:
			conf = getattr(self, self.configuration)
			conf()

		except: 
			print("Can not find configuration")
			raise
			
			flags.DEFINE_string("mode", "normal", "operation mode: normal or freq [normal]")


	def example(self):

		train_config = self.config["train"]

		train_config["mode"] = "normal" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 10000  # Number of epoch [10000]
		train_config["batch_size"] = 128 # The size of batch images [128]
		train_config["image_size"] = 32 # The size of image to use [33]
		train_config["label_size"] = 20 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 1 # Dimension of image color. [1]
		train_config["scale"] = 4 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["test_extract_stride"] # The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "checkpoint" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "log"
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "preprocess/output" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "grr_grid_srcnn_v1" # Name of checkpoints

		def srcnn(self):
						
			mconfig = {}
			mconfig["grr_grid_srcnn_v1"] = {

										"scale":[4],
										"subimages":[20,20],
										"padding":[6,6],
										"ckpt_file":"/home/ubuntu/model/model/SR_project/SRCNN/SRCNN.model-309672",
										"isGray": True,
										"isNormallized":True,
										"upsample": False
										}
			return mconfig


		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = './preprocess/Test'
		eval_config["test_set"] = ["Set5", "Set14", "BSD100"]
		eval_config["models"] = [srcnn(self)]
		eval_config["summary_file"] = "example_summary.txt"

	# Autoencoder model config
	def AE_baseline_model(self):

		####################################
		#             Training             #
		####################################        
		train_config = self.config["train"]

		train_config["mode"] = "autoencoder" 
		train_config["iteration"] = 12000000 
		train_config["curr_iteration"] = 0
		train_config["batch_size"] = 16 
		train_config["image_size"] = 256
#		train_config["project_image_size"] = 2 * round( math.sqrt((train_config["image_size"]/2)**2 + (train_config["image_size"]/2)**2) ) + 1
#		train_config["project_image_size"] = 384        
		train_config["project_image_size"] = [30,363]
		train_config["learning_rate"] = 1e-4
		train_config["checkpoint_dir"] = file_root + "model/MDetection/autoencoder_model" 
		train_config["log_dir"] = file_root + "model/MDetection/autoencoder_model/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
#		train_config["ckpt_name"] = "baseline_v5_encdec_flatten_batch_16_code_64x64"        
#		train_config["ckpt_name"] = "baseline_v5_encdec_flatten_batch_16_code_64x64_numlines_30"                

#		train_config["ckpt_name"] = "baseline_v6_encdec_flatten_batch_16_code_64x64_numlines_30"        
#		train_config["ckpt_name"] = "baseline_v6_encdec_flatten_batch_16_code_32x32_numlines_30"                 

#		train_config["ckpt_name"] = "baseline_end2end_encdec_batch_16_code_64x64"                 
#		train_config["ckpt_name"] = "baseline_end2end_2D_encdec_batch_16_code_16x16x16"                         
#		train_config["ckpt_name"] = "baseline_end2end_2D_encdec_batch_16_code_16x16x16_normalized"                                 
#		train_config["ckpt_name"] = "baseline_end2end_2D_v2_encdec_batch_16_code_16x16x16"         
#		train_config["ckpt_name"] = "baseline_end2end_2D_v2_encdec_batch_16_code_1024"                 
#		train_config["ckpt_name"] = "baseline_end2end_2D_v2_encdec_batch_16_code_4096"                
#		train_config["ckpt_name"] = "baseline_end2end_2D_v2_encdec_batch_16_code_conv_16x16x16"                
		train_config["ckpt_name"] = "baseline_end2end_2D_v2_encdec_batch_16_code_conv_8x8x16"              

		train_config["is_train"] = True # True for training, False for testing [True]
#		train_config["model_ticket"] = "baseline" # Name of checkpoints
#		train_config["model_ticket"] = "baseline_v2" # Name of checkpoints        
#		train_config["model_ticket"] = "baseline_v3" # Name of checkpoints                
#		train_config["model_ticket"] = "baseline_v4" # Name of checkpoints                          
#		train_config["model_ticket"] = "baseline_v5" # Name of checkpoints                                  
#		train_config["model_ticket"] = "baseline_v5_flatten" # Name of checkpoints                                          

#		train_config["model_ticket"] = "baseline_v6_flatten" # Name of checkpoints          

#		train_config["model_ticket"] = "baseline_end2end" # Name of checkpoints          
#		train_config["model_ticket"] = "baseline_end2end_2D" # Name of checkpoints                 
		train_config["model_ticket"] = "baseline_end2end_2D_v2" # Name of checkpoints                         
        
		train_config["inputroot"] = file_root + 'dataset/MDetection/ICPR2012/testing_data/scanner_A/bmp/'        
		train_config["labelroot"] = file_root + 'dataset/MDetection/ICPR2012/testing_data/scanner_A/label_plot/'# for the input of the encoder         
        
		####################################
		#            Evaluation            #
		####################################

		eval_config = self.config["evaluation"]

		eval_config["mode"] = 'encode'        
#		eval_config["mode"] = 'decode'                
        
		eval_config["sample_mode"] = 'split'                
#		eval_config["sample_mode"] = 'random'                        
        
#		eval_config["aug"] = True   
		eval_config["aug"] = False                                

		eval_config["image_size"] = 256
		eval_config["project_image_size"] = [30,363]
#		eval_config["code_size"] = [64,64,1]       
#		eval_config["code_size"] = [64,64,1]               
		eval_config["code_size"] = [16,16,16]                       
#		eval_config["model_ticket"] = "baseline_v5_flatten" 
#		eval_config["model_ticket"] = "baseline_v6_flatten"  
#		eval_config["model_ticket"] = "baseline_end2end"          
#		eval_config["model_ticket"] = "baseline_end2end_2D"                  
		eval_config["model_ticket"] = "baseline_end2end_2D_v2"            
         
#		eval_config["ckpt_file"] = file_root + "model/MDetection/autoencoder_model/Temp/baseline_v5_encdec_flatten_batch_16_code_64x64_numlines_30_0.053725466132164-1370000" 
#		eval_config["ckpt_file"] = file_root + "model/MDetection/autoencoder_model/Temp/baseline_v5_encdec_flatten_batch_16_code_64x64_numlines_30_0.12017565220594406-494000"         
#		eval_config["ckpt_file"] = file_root + "model/MDetection/autoencoder_model/Temp/baseline_v6_encdec_flatten_batch_16_code_64x64_numlines_30_0.06631352007389069-463000" 
#		eval_config["ckpt_file"] = file_root + "model/MDetection/autoencoder_model/Temp/baseline_end2end_encdec_batch_16_code_64x64_0.003570693312212825-214000" 
#		eval_config["ckpt_file"] = file_root + "model/MDetection/autoencoder_model/Temp/baseline_end2end_2D_encdec_batch_16_code_16x16x16_0.00727227795869112-57000"         
#		eval_config["ckpt_file"] = file_root + "model/MDetection/autoencoder_model/Temp/baseline_end2end_2D_v2_encdec_batch_16_code_16x16x16_0.0004004873044323176-743000"         
#		eval_config["ckpt_file"] = file_root + "model/MDetection/autoencoder_model/Temp/baseline_end2end_2D_v2_encdec_batch_16_code_4096_0.004421569406986237-429000"                 
#		eval_config["ckpt_file"] = file_root + "model/MDetection/autoencoder_model/Temp/baseline_end2end_2D_v2_encdec_batch_16_code_conv_4096_0.0029130112379789352-455000"                         
		eval_config["ckpt_file"] = file_root + "model/MDetection/autoencoder_model/Temp/baseline_end2end_2D_v2_encdec_batch_16_code_conv_16x16x16_0.0006160975317470729-200000"                                 
        
#		eval_config["inputroot"] = '/data/wei/dataset/MDetection/ICPR2012/training_data/scanner_A/bmp/'        
#		eval_config["labelroot"] = '/data/wei/dataset/MDetection/ICPR2012/training_data/scanner_A/label_plot/'# for the input of the encoder         
#		eval_config["coderoot"] = '/data/wei/dataset/MDetection/ICPR2012/training_data/scanner_A/label_code/' + str(eval_config["image_size"]) + 'x' + str(eval_config["image_size"]) + '/'# for the output of the encoder & the input of the detection  
        
		eval_config["inputroot"] = file_root + 'dataset/MDetection/ICPR2012/' + data_type + '/scanner_A/bmp/'        
		eval_config["labelroot"] = file_root + 'dataset/MDetection/ICPR2012/' + data_type + '/scanner_A/label_plot/'# for the input of the encoder         
		eval_config["coderoot"] =  file_root + 'dataset/MDetection/ICPR2012/' + data_type + '/scanner_A/label_code/' + str(eval_config["image_size"]) + 'x' + str(eval_config["image_size"]) + '/'# for the output of the encoder & the input of the detection   

		eval_config["predroot"] =  file_root + 'dataset/MDetection/ICPR2012/' + data_type + '/scanner_A/label_pred/' # for the input of the decoder from detection     
     
		eval_config["enc_output_dir"] = eval_config["coderoot"]                
		eval_config["dec_output_dir"] = "./dec_output/" 

	# Detection model config
	def DE_baseline_model(self):

		####################################
		#             Training             #
		####################################      
		train_config = self.config["train"]

		train_config["mode"] = "detection" 
		train_config["iteration"] = 800000 
		train_config["curr_iteration"] = 0
		train_config["batch_size"] = 32 
		train_config["image_size"] = 256           
		train_config["project_image_size"] = [30,363]
#		train_config["code_size"] = [64,64,1]           
#		train_config["code_size"] = [32,32,1]          
		train_config["code_size"] = [16,16,16]                  
		train_config["learning_rate"] = 1e-4
		train_config["checkpoint_dir"] = file_root + "model/MDetection/detection_model" 
		train_config["log_dir"] = file_root + "model/MDetection/detection_model/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]             
#		train_config["ckpt_name"] = "DE_alexnet_nab_13_normalized"                 
#		train_config["ckpt_name"] = "DE_alexnet_nab_19_normalized_04encoder"                 
#		train_config["ckpt_name"] = "DE_alexnet_nab_13_normalized_04encoder"                   
#		train_config["ckpt_name"] = "DE_alexnet_split_normalized_04encoder"              
#		train_config["ckpt_name"] = "DE_baseline_nab_13_normalized"            
#		train_config["ckpt_name"] = "DE_baseline_nab_13_normalized_04encoder"                    
#		train_config["ckpt_name"] = "DE_baseline_nab_19_normalized_04encoder"                            
#		train_config["ckpt_name"] = "DE_baseline_normalized109_1024encoder_005_l2reg_adaptive_ratio_05"                  
#		train_config["ckpt_name"] = "DE_baseline_1024encoder_005_AbTest"          
#		train_config["ckpt_name"] = "DE_baseline_2D_16x16x16_end2end_2D_encoder_0007_normalized255_aug_L1_Xavier"        
        
#		train_config["ckpt_name"] = "DE_alexnet_nab_13_normalized_1024encoder"                         
#		train_config["ckpt_name"] = "DE_alexnet_nab_13_normalized_1024encoder_l2reg"               
#		train_config["ckpt_name"] = "DE_alexnet_normalized_1024encoder_l2reg"                       
#		train_config["ckpt_name"] = "DE_alexnet_1024encoder_005_AbTest"             
#		train_config["ckpt_name"] = "DE_alexnet_1024encoder_005_adaptive_ratio_05"             
#		train_config["ckpt_name"] = "DE_alexnet_normalized_1024encoder_005_l2reg"                               
#		train_config["ckpt_name"] = "DE_alexnet_normalized_1024encoder_005_l2reg_adaptive_ratio"                        
#		train_config["ckpt_name"] = "DE_alexnet_normalized255_1024encoder_005_l2reg_adaptive_ratio_05"            
#		train_config["ckpt_name"] = "DE_alexnet_normalized109_1024encoder_005_l2reg_adaptive_ratio_05"                   
#		train_config["ckpt_name"] = "DE_alexnet_normalized109_1024encoder_005_l2reg_AbTest"     

#		train_config["ckpt_name"] = "DE_alexnet_normalized56_4096encoder_006_l2reg_AbTest"                         
#		train_config["ckpt_name"] = "DE_alexnet_4096encoder_006_l2reg_AbTest"    
#		train_config["ckpt_name"] = "DE_alexnet_4096encoder_006_AbTest"            

#		train_config["ckpt_name"] = "DE_alexnet_4096_end2end_encoder_0003_AbTest"            
#		train_config["ckpt_name"] = "DE_alexnet_4096_end2end_encoder_0003_dropout_AbTest"              
#		train_config["ckpt_name"] = "DE_alexnet_4096_end2end_encoder_0003_vali_AbTest"             
#		train_config["ckpt_name"] = "DE_alexnet_4096_end2end_encoder_0004"            
#		train_config["ckpt_name"] = "DE_alexnet_conv_4096_end2end_encoder_0004"                    
		train_config["ckpt_name"] = "DE_alexnet_conv_16x16x16_end2end_encoder_0004_AbTest"                  
        
#		train_config["ckpt_name"] = "DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007_vali_AbTest"        
#		train_config["ckpt_name"] = "DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007_AbTest"               
#		train_config["ckpt_name"] = "DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007"                
#		train_config["ckpt_name"] = "DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007_normalized"              
#		train_config["ckpt_name"] = "DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007_normalized255"              
#		train_config["ckpt_name"] = "DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007_normalized255_aug"            
#		train_config["ckpt_name"] = "DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007_normalized255_aug_91"             
#		train_config["ckpt_name"] = "DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007_normalized255_aug_L1"          
#		train_config["ckpt_name"] = "DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007_normalized255_aug_L1_Xavier"                  
#		train_config["ckpt_name"] = "DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007_normalized255_aug_momentum"                     
        
		train_config["is_train"] = True # True for training, False for testing [True]
#		train_config["model_ticket"] = "baseline" # Name of checkpoints
#		train_config["model_ticket"] = "alex_net" # Name of checkpoints                
		train_config["model_ticket"] = "alex_net_2D" # Name of checkpoints           
#		train_config["model_ticket"] = "baseline_2D" # Name of checkpoints      
        
		train_config["train_code_root"] = file_root + 'dataset/MDetection/ICPR2012/training_data/scanner_A/label_code/' + str(train_config["image_size"]) + 'x' + str(train_config["image_size"]) + '/'        
		train_config["test_code_root"] = file_root + 'dataset/MDetection/ICPR2012/testing_data/scanner_A/label_code/' + str(train_config["image_size"]) + 'x' + str(train_config["image_size"]) + '/'        

		train_config["train_root"] = file_root + 'dataset/MDetection/ICPR2012/training_data/scanner_A/classfied_data/' + str(train_config["image_size"]) + 'x' + str(train_config["image_size"]) + '/'        
		train_config["test_root"] = file_root + 'dataset/MDetection/ICPR2012/testing_data/scanner_A/classfied_data/' + str(train_config["image_size"]) + 'x' + str(train_config["image_size"]) + '/'        
        
		####################################
		#            Evaluation            #
		####################################
		eval_config = self.config["evaluation"]
        
		eval_config["mode"] = 'detection'         
                  
		eval_config["image_size"] = 256
		eval_config["project_image_size"] = [30,363]
#		eval_config["code_size"] = [32,32,1]       
		eval_config["code_size"] = [16,16,16]               
        
#		eval_config["model_ticket"] = "baseline" # Name of checkpoints
#		eval_config["model_ticket"] = "alex_net" 
		eval_config["model_ticket"] = "alex_net_2D"         

#		eval_config["ckpt_file"] = file_root + "model/MDetection/detection_model/Temp/DE_baseline_normalized109_1024encoder_005_l2reg_adaptive_ratio_05_6.987153756199405e-05-88000" 
		eval_config["ckpt_file"] = file_root + "model/MDetection/detection_model/Temp/DE_alexnet_2D_16x16x16_end2end_2D_encoder_0007_normalized255_aug_L1_4.7751288414001465-46000" 

		eval_config["inputroot"] = file_root + 'dataset/MDetection/ICPR2012/testing_data/scanner_A/bmp/'        
		eval_config["labelroot"] = file_root + 'dataset/MDetection/ICPR2012/testing_data/scanner_A/label_plot/'# for the input of the encoder        
		eval_config["coderoot"] =  file_root + 'dataset/MDetection/ICPR2012/testing_data/scanner_A/label_code/' + str(eval_config["image_size"]) + 'x' + str(eval_config["image_size"]) + '/'# for the output of the encoder & the input of the detection        

		eval_config["predroot"] =  file_root + 'dataset/MDetection/ICPR2012/testing_data/scanner_A/label_pred/' # for the input of the decoder from detection       
        
		eval_config["detect_output_dir"] = "./detect_output/" 
        
		####################################
		#            Preprocess            #
		####################################              
		prpc_config = self.config["preprocess"]
      
		prpc_config["coderoot"] =  file_root + 'dataset/MDetection/ICPR2012/' + data_type + '/scanner_A/label_code/' + str(eval_config["image_size"]) + 'x' + str(eval_config["image_size"]) + '/'# for the output of the encoder & the input of the detection              
		prpc_config["output_dir"] =  file_root + 'dataset/MDetection/ICPR2012/' + data_type + '/scanner_A/classfied_data/' + str(eval_config["image_size"]) + 'x' + str(eval_config["image_size"]) + '/'# for the output of the encoder & the input of the detection                

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
