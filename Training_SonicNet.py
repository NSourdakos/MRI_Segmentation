from Dataset import *
from Utils import *
from SonicNet import *
from Unet_PlusPlus import *
import tensorflow as tf
#	python G:\Books\AI\CODE\Internship\Training_SonicNet.py
import os
from tensorflow.python.client import device_lib


def calculate_train_samples(data_shape,data_cube_size,data_overlap_percentage):
	'''
	This function calculates the number of samples sent to the generator

	Inputs:

	data_shape = the shape of the data, here given to the function by the .json
				config file
	data_cube_size = the size of the cube of each sample. Here only the first
					dimension is counted, as it is assumed that we train with cubes.
	data_overlap_percentage = the percentage the different samples overlap with each other.
	'''
	data_new_size = int(data_cube_size*(1-data_overlap_percentage))
	Counter_X_Limit = int((data_shape[0] - data_cube_size)//data_new_size)
	Counter_Y_Limit = int((data_shape[1] - data_cube_size)//data_new_size)
	Counter_Z_Limit = int((data_shape[2] - data_cube_size)//data_new_size)
	

	  #Case for not Integer
	if data_shape[0]> Counter_X_Limit*data_new_size + data_cube_size:
		#print("NOT INTEGER X")
	  
	  Counter_X_Limit +=1
	if data_shape[1]> Counter_Y_Limit*data_new_size + data_cube_size:
		#print("NOT INTEGER Y")
	  
	  Counter_Y_Limit +=1
	if data_shape[2]> Counter_Z_Limit*data_new_size + data_cube_size:
		#print("NOT INTEGER Z")
	  
	  Counter_Z_Limit +=1

	total_train_size = (Counter_X_Limit+1)*(Counter_Y_Limit+1)*(Counter_Z_Limit+1)#Account for begining at 0 with a +1
	return total_train_size


def training(config):

	'''
	Main training function. It instantiates the model, calculates the training batches, instantiates the dataset
	and the data generators, and finally calls the function that does the fitting.
	'''

	model = SonicNet(config) #Also hard coded

	print(device_lib.list_local_devices())
	#tf.test.gpu_device_name()
	data_cube_size = config['model']["input_size"][0]
	data_shape = config['dataset']['shape']
	data_overlap_percentage = config['dataset']['overlap_percentage']
	#num_train_samples = config['training']['num_train_samples']
	#num_test_samples = config['training']['num_test_samples']

 
	num_train_samples = calculate_train_samples(data_shape,data_cube_size,data_overlap_percentage)
	num_test_samples = calculate_train_samples(data_shape,data_cube_size,data_overlap_percentage)

	print("Number of training samples is:", num_train_samples)

	dataset=FreeSurfer_Dataset(config, model).load_dataset_2()
	train_set_generator = dataset.createDataGenerator('train')
	test_set_generator = dataset.createDataGenerator('test')
	
	model.fit_model(train_set_generator, num_train_samples, test_set_generator, num_test_samples,
						  config['training']['num_epochs'])
	



if __name__ == '__main__':


	#These two are hard-coded for now

	plt = platform.system()
	if plt == "Windows":
		slash = "\\"
	else:
		slash = "/"

	model_path = slash+"Models"+slash+"SonicNet"+slash
	config_name = "ConfigMRI_SonicNet.json"


	set_system_settings()
	base_path = os.path.dirname(os.path.realpath(__file__))
	print(base_path)
		
	config_path = base_path + model_path + config_name
	config = load_config(config_path)
	print(config)
	#Correction of the path variables in config dictionary
	config['dataset']['path'] = base_path + slash+"Results"+slash
	config['dataset']['dictionary_path'] = base_path + slash+"label_convert_final.npy"
	config['training']['path'] = base_path + model_path+"Training"+slash

	#print(config)




	training(config)
