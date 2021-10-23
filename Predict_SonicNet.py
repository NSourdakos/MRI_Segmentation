from Dataset import *
from Utils import *
from SonicNet import *

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
#import nibabel as nb
from sklearn.metrics import f1_score
import platform

import os
from tensorflow.python.client import device_lib

def calculate_data_limits(data_shape,data_cube_size):
  '''
  This function calculates the number of samples sent to the generator

  Inputs:

  data_shape = the shape of the data, here given to the function by the .json
        config file
  data_cube_size = the size of the cube of each sample. Here only the first
          dimension is counted, as it is assumed that we train with cubes.
  data_overlap_percentage = 0.5
  '''


  
  data_new_size = int(data_cube_size*(1-0.5))
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
  return Counter_X_Limit,Counter_Y_Limit,Counter_Z_Limit


#FULL PREDICT




def fix_labels(labels,config):

  '''
  This function fixes the labels from the original
  Freesurfer to the 30 labels that we use
  '''
  label_path = config['dataset']['dictionary_path']
  label_conv_matrix = np.load(label_path)
  label_dict = dict(zip(label_conv_matrix[:,0], label_conv_matrix[:,1]))
  labels_new = np.zeros_like(labels)
  

  
  for x in range(labels.shape[0]):
    for y in range(labels.shape[1]):
      for z in range(labels.shape[2]):
          categ = int(labels[x,y,z])
          if (categ in label_dict.keys()):
            new_categ=label_dict[categ]
            labels_new[x,y,z] = new_categ
  return labels_new



def prediction(config):

  '''
  This function uses the pretrained model to predict the labels

  left_out_data is a parameter that separates the test set from the train set.
        It is thus set to 40

  subject_list finds all the files in the dataset folder, and returns their names in a list




  OUTPUTS:

  predicted_output_matrices: All the predicted brain label matrices
  '''
  plt = platform.system()
  if plt == "Windows":
    slash = "\\"
  else:
    slash = "/"  

  output_acc = (config['training']["path"]+"Predicted_Accuracies.npy")
  
  label_matrix_file = config['dataset']['dictionary_path']
  label_conv_matrix = np.load(label_matrix_file)
  label_conv_matrix_int  = label_conv_matrix.astype(int)

  #THIS LABEL_DICT IS USED TO PROJECT BACK TO THE FREESURFER LABEL_SPACE. CURRENTLY NOT IN USE.
  label_dict = { 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 24, 19: 26, 20: 27,  21: 28, 22: 31, 23: 72, 24: 77, 25: 85, 26: 192, 27: 98, 28: 30, 29: 251}


  data_cube_size = config['model']["input_size"][0]
  data_shape = config['dataset']['shape']
  data_overlap_percentage = config['dataset']['overlap_percentage']

  base_dir = config['dataset']['path']
  subject_list = os.listdir(base_dir)
  subject_list.sort()
  left_out_data = config['dataset']['test_data']


  model = SonicNet(config)
  
  Accuracies = np.zeros((left_out_data,1))
  counter = 0
  output_acc = (config['training']["path"]+"Predicted_Accuracies.npy")
  output_predicted = (config['training']["path"]+"Predicted_Labels.npz")
  outputs = []
  for subject in subject_list:
  

    #This part needs modification, to load other data. Allocate the new data on a numpy format to data_X_Full, and everything should work fine
    #Data also needs to be normalized between 0 and 1.
    print("Testing on subject:", subject)
    loaded_X = np.load(base_dir+ subject +"/Data_X.npz")

    data_X_Full = loaded_X['arr_0']

  
    output = np.zeros_like(data_X_Full)
    Counter_X_Limit,Counter_Y_Limit,Counter_Z_Limit = calculate_data_limits(data_X_Full.shape,data_cube_size)
    Total_counter=0
    for X in range(0, (Counter_X_Limit + 1),1):
    
      
      for Y in range(0, (Counter_Y_Limit + 1),1):
      
       
        for Z in range(0, (Counter_Z_Limit + 1 ),1):
          Total_counter +=1
          total_predict_boxes=Counter_X_Limit*Counter_Y_Limit*Counter_Z_Limit
          print("Predicting on volume", Total_counter,"out of", total_predict_boxes,".")
        
          half_data_cube = data_cube_size//2

          if X==Counter_X_Limit:
            left_B =data_shape[0] - data_cube_size  
          else:
            left_B = X*half_data_cube
           
          right_B = left_B + data_cube_size 

          if Y==Counter_Y_Limit:
            back_B =data_shape[1] - data_cube_size
          else:
            back_B = Y*half_data_cube 
          front_B = back_B + data_cube_size

          if Z==(Counter_Z_Limit):
            bottom_B =data_shape[2] - data_cube_size
          else:
            bottom_B = Z*half_data_cube 
          #print(bottom_B)
          top_B = bottom_B  + data_cube_size
        


          data_X = data_X_Full[left_B:right_B,back_B:front_B,bottom_B:top_B]
          data_X = np.expand_dims(data_X, axis=0)
          data_X = np.expand_dims(data_X, axis=4)


        
          pred = model.predict(data_X)
          Y_pred=np.argmax(pred, axis=4)
          Y_pred_Sq = np.squeeze(Y_pred)
          Y_pred_Final = np.zeros_like(Y_pred_Sq)
          for x in range(Y_pred_Sq.shape[0]):
            for y in range(Y_pred_Sq.shape[1]):
              for z in range(Y_pred_Sq.shape[2]):
                if (int(Y_pred_Sq[x,y,z]) in label_dict.keys()):
                  #Y_pred_Final[x,y,z] = int(label_dict[Y_pred_Sq[x,y,z]])#CHANGED FROM FREESURFER LABEL SPACE TO OUR OWN
                  Y_pred_Final[x,y,z] = int(Y_pred_Sq[x,y,z])
        
          output[left_B:right_B,back_B:front_B,bottom_B:top_B] = Y_pred_Final

    output_predicted_path = (config['training']['output_path']+slash +"Predicted_Labels_"+str(counter)+".npz")
    #outputs.append(output)
    np.save(output_predicted_path, output, allow_pickle=True)

    counter += 1
    
    
    loaded_X.close()
    #loaded_Y.close()
    del loaded_X
    #del loaded_Y
    gc.collect()

  predicted_output_matrices = np.asarray(outputs)
  np.savez_compressed(output_predicted,predicted_output_matrices)
  #np.save(output_acc, Accuracies, allow_pickle=True)
  #print("Average test accuracy is:",np.mean(Accuracies))
  #print("FINALLY DONE")
  


if __name__ == '__main__':

  plt = platform.system()
  if plt == "Windows":
    slash = "\\"
  else:
    slash = "/"  


  #These two are hard-coded for now
  model_path = slash +"Models"+ slash + "SonicNet" +slash
  config_name = "ConfigMRI_SonicNet.json"


  set_system_settings()
  base_path = os.path.dirname(os.path.realpath(__file__))
  print(base_path)
    
  config_path = base_path + model_path+config_name
  config = load_config(config_path)
  
  #Correction of the path variables in config dictionary
  config['dataset']['path'] = base_path + slash+ "Prepro_Data" + slash
  config['dataset']['dictionary_path'] = base_path + slash+"label_convert_final.npy"
  config['training']['path'] = base_path + model_path+"Training" + slash
  config['training']['output_path'] = base_path  + slash + "Output_Data"


  




  prediction(config)
