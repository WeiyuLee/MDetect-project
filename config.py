import math

class config:

	def __init__(self, configuration):
		
		self.configuration = configuration
		self.config = {
						"common":{},
						"train":{},
						"evaluation":{
								"dataroot":None,
								"test_set":["Set5", "Set14", "BSD100"],
								"models":{},
								
							}
						}
		self.get_config()


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
		train_config["iteration"] = 8000000 
		train_config["curr_iteration"] = 4783501
		train_config["batch_size"] = 16 
		train_config["image_size"] = 256
#		train_config["project_image_size"] = 2 * round( math.sqrt((train_config["image_size"]/2)**2 + (train_config["image_size"]/2)**2) ) + 1
		train_config["project_image_size"] = 384        
		train_config["learning_rate"] = 1e-4/2 
		train_config["checkpoint_dir"] = "/home/sdc1/model/MDetection/autoencoder_model" 
		train_config["log_dir"] = "/home/sdc1/model/MDetection/autoencoder_model/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
#		train_config["ckpt_name"] = "test_v2"
#		train_config["ckpt_name"] = "test_v2_max_32"        
#		train_config["ckpt_name"] = "test_v3_max_32_decode"                
#		train_config["ckpt_name"] = "test_v3_max_5_decode_256"        
#		train_config["ckpt_name"] = "baseline_v3_max_5_decode_256"                
#		train_config["ckpt_name"] = "baseline_v3_max_5_decode_256_code_2"              
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_8"                      
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_32"               
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_4"         
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_1"          
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_2"            
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_2_lrelu"         
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_4_lrelu"        
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_128_code_4"                
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_4_res_32_relu_L2"                 
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_4_res_32_lrelu_L2"                
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_4_res_16_lrelu_L2"           
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_4_res_16_lrelu_L2_test_2"                   
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_4_res_16_lrelu_L2_test_4"                           
#		train_config["ckpt_name"] = "baseline_v4_max_32_decode_256_code_4_res_16_lrelu_L2_test_4"          
#		train_config["ckpt_name"] = "baseline_v4_max_32_decode_256_code_4_res_16_lrelu_L2_test_2_encdec"                  
#		train_config["ckpt_name"] = "baseline_v4_max_5_decode_256_code_8_res_16_lrelu_L2_test_2_encdec"                 
#		train_config["ckpt_name"] = "baseline_v4_encdec"             
#		train_config["ckpt_name"] = "baseline_v4_encdec_64x64"                     
#		train_config["ckpt_name"] = "baseline_v5_encdec"           
#		train_config["ckpt_name"] = "baseline_v5_encdec_64x64"            
#		train_config["ckpt_name"] = "baseline_v5_encdec_64x64_relu"        
#		train_config["ckpt_name"] = "baseline_v5_encdec_flatten"          
		train_config["ckpt_name"] = "baseline_v5_encdec_flatten_batch_16_code_64x64"        
		train_config["is_train"] = True # True for training, False for testing [True]
#		train_config["model_ticket"] = "baseline" # Name of checkpoints
#		train_config["model_ticket"] = "baseline_v2" # Name of checkpoints        
#		train_config["model_ticket"] = "baseline_v3" # Name of checkpoints                
#		train_config["model_ticket"] = "baseline_v4" # Name of checkpoints                          
#		train_config["model_ticket"] = "baseline_v5" # Name of checkpoints                                  
		train_config["model_ticket"] = "baseline_v5_flatten" # Name of checkpoints                                          

		####################################
		#            Evaluation            #
		####################################

		eval_config = self.config["evaluation"]

		eval_config["mode"] = 'encode'        
#		eval_config["mode"] = 'decode'                
        
#		eval_config["sample_mode"] = 'split'                
		eval_config["sample_mode"] = 'random'                        
        
		eval_config["aug"] = True   
#		eval_config["aug"] = False                                

		eval_config["image_size"] = 224
		eval_config["project_image_size"] = 384
		eval_config["code_size"] = [64,64,1]       
		eval_config["model_ticket"] = "baseline_v5_flatten" 
		eval_config["ckpt_file"] = "/home/sdc1/model/MDetection/autoencoder_model/Temp/baseline_v5_encdec_flatten_batch_16_code_64x64-1003250" 

		eval_config["inputroot"] = '/home/sdc1/dataset/ICPR2012/training_data/scanner_A/bmp/'        
		eval_config["labelroot"] = '/home/sdc1/dataset/ICPR2012/training_data/scanner_A/label_plot/'# for the input of the encoder         
		eval_config["coderoot"] = '/home/sdc1/dataset/ICPR2012/training_data/scanner_A/label_code/' + str(eval_config["image_size"]) + 'x' + str(eval_config["image_size"])# for the output of the encoder & the input of the detection  
        
#		eval_config["inputroot"] = '/home/sdc1/dataset/ICPR2012/testing_data/scanner_A/bmp/'        
#		eval_config["labelroot"] = '/home/sdc1/dataset/ICPR2012/testing_data/scanner_A/label_plot/'# for the input of the encoder         
#		eval_config["coderoot"] = '/home/sdc1/dataset/ICPR2012/testing_data/scanner_A/label_code/' + str(eval_config["image_size"]) + 'x' + str(eval_config["image_size"])# for the output of the encoder & the input of the detection   

		eval_config["predroot"] = '/home/sdc1/dataset/ICPR2012/testing_data/scanner_A/label_pred/' # for the input of the decoder from detection     
     
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
		train_config["curr_iteration"] = 111001
		train_config["batch_size"] = 16 
		train_config["image_size"] = 256           
		train_config["project_image_size"] = 384
		train_config["code_size"] = [64,64,1]           
		train_config["learning_rate"] = 1e-4 
		train_config["checkpoint_dir"] = "/home/sdc1/model/MDetection/autoencoder_model" 
		train_config["log_dir"] = "/home/sdc1/model/MDetection/autoencoder_model/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]             
#		train_config["ckpt_name"] = "DE_baseline"                 
#		train_config["ckpt_name"] = "DE_baseline_flatten"            
#		train_config["ckpt_name"] = "DE_baseline_flatten_aug"         
#		train_config["ckpt_name"] = "DE_baseline_flatten_aug_alex_ent"                 
#		train_config["ckpt_name"] = "DE_baseline_flatten_aug_alex_net_nab_13"           
		train_config["ckpt_name"] = "DE_baseline_flatten_aug_alex_net_random_crop_nab_13" 
#		train_config["ckpt_name"] = "DE_baseline_flatten_aug_alex_net_random_crop_nab_11"                  
#		train_config["ckpt_name"] = "DE_baseline_flatten_aug_random_crop_nab_11"                 
		train_config["is_train"] = True # True for training, False for testing [True]
#		train_config["model_ticket"] = "baseline" # Name of checkpoints
		train_config["model_ticket"] = "alex_net" # Name of checkpoints        
		train_config["datasetroot"] = '/home/sdc1/dataset/ICPR2012/training_data/scanner_A/classfied_data/' + str(train_config["image_size"]) + 'x' + str(train_config["image_size"])        

		####################################
		#            Evaluation            #
		####################################
		eval_config = self.config["evaluation"]
        
		eval_config["mode"] = 'detection'         
                  
		eval_config["image_size"] = 256
		eval_config["project_image_size"] = 384
		eval_config["code_size"] = [64,64,1]       
		eval_config["model_ticket"] = "alex_net" 
		eval_config["ckpt_file"] = "/home/sdc1/model/MDetection/detection_model/Temp/DE_baseline_flatten_aug_alex_net_random_crop_nab_13-416000" 

		eval_config["inputroot"] = '/home/sdc1/dataset/ICPR2012/testing_data/scanner_A/bmp/'        
		eval_config["labelroot"] = '/home/sdc1/dataset/ICPR2012/testing_data/scanner_A/label_plot/'# for the input of the encoder        
		eval_config["coderoot"] = '/home/sdc1/dataset/ICPR2012/testing_data/scanner_A/label_code/' + str(eval_config["image_size"]) + 'x' + str(eval_config["image_size"])# for the output of the encoder & the input of the detection        

		eval_config["predroot"] = '/home/sdc1/dataset/ICPR2012/testing_data/scanner_A/label_pred/' # for the input of the decoder from detection       
        
		eval_config["detect_output_dir"] = "./detect_output/" 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
