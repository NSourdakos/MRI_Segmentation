from Dataset import *
from Utils import *
from SonicNet import *
#from Unet_PlusPlus import *
import tensorflow as tf

import os
from tensorflow.python.client import device_lib


import os
import time
import csv
import numpy as np
import nibabel as nb


import matplotlib.pyplot as plt
def find_standard_boundaries(left_B,right_B,back_B,front_B,bottom_B,top_B,max_Del_X,max_Del_Y,max_Del_Z):
  """
  This function pads each brain with zeros, so that its dimensions are the same as the ones
  passed by the parameters max_Del_X, max_Del_Y and max_Del_Z
  """
  if right_B - left_B < max_Del_X:
    counter = 0
    while right_B - left_B < max_Del_X:
      if counter % 2==0 and left_B>0:
        left_B -=1
      else:
        right_B +=1
      counter +=1
  elif right_B - left_B > max_Del_X:
    counter = 0
    while right_B - left_B > max_Del_X:
      if counter % 2==0:
        left_B +=1
      else:
        right_B -=1
      counter +=1
  #Make everything the same size Y by adding zeros
  if front_B - back_B < max_Del_Y:
    counter = 0
    while front_B - back_B < max_Del_Y:
      if counter % 2==0 and back_B>0:
        back_B -=1
      else:
        front_B +=1
      counter +=1
  elif front_B - back_B > max_Del_Y:
    counter = 0
    while front_B - back_B > max_Del_Y:
      if counter % 2==0:
        back_B +=1
      else:
        front_B -=1
      counter +=1
  #Make everything the same size Z by adding zeros
  if top_B - bottom_B < max_Del_Z:
    counter = 0
    while top_B - bottom_B < max_Del_Z:
      if counter % 2==0 and bottom_B>0:
        bottom_B -=1
      else:
        top_B +=1
      counter +=1
  elif top_B - bottom_B > max_Del_Z:
    counter = 0
    while top_B - bottom_B > max_Del_Z:
      if counter % 2==0:
        bottom_B +=1
      else:
        top_B -=1
      counter +=1 
  return left_B,right_B,back_B,front_B,bottom_B,top_B

def func_nrm_range_zero_one(ary_ima, max=255, dtype=np.float32):
    """Normalize image to be in range from 0 to 1.

    Parameters
    ----------
    ary_ima : numpy array
        Array with the image data that should be normalized.
    dtype : type, one from np.float32 or np.int32
        Datatype that the input array should be set to.

    Returns
    -------
    ary_nrm_ima : numpy array
        Normalized image data in range from 0 to 1.

    """
    # Make sure that data has correct type
    ary_ima = ary_ima.astype(np.float32)
    # Get minimum of the image array
    var_min = 0
    # Get maximum of the image array
    var_max = 255
    # Bring image data into range from 0 to 1
    if var_max > var_min:
        ary_nrm_ima = np.divide(np.subtract(ary_ima, var_min),
                                np.subtract(var_max, var_min))
    else:
        ary_nrm_ima = np.multiply(ary_ima, 0.)

    return ary_nrm_ima

def find_boundaries(data_norm):
  """This function finds the boundaries of each brain"""
  left_B = 0
  right_B = data_norm.shape[0]-1
  back_B = 0
  front_B = data_norm.shape[1]-1
  bottom_B = 0
  top_B = data_norm.shape[2]-1
  
  for x in range(data_norm.shape[0]//2):
    if np.all(data_norm[x,:,:]==0):
      left_B = x

  for x in range(data_norm.shape[0]//2):
    if np.all(data_norm[-x,:,:]==0):
      right_B =data_norm.shape[0]- x

  for y in range(data_norm.shape[1]//2):
    if np.all(data_norm[:,y,:]==0):
      back_B = y

  for y in range(data_norm.shape[1]//2):
    if np.all(data_norm[:,-y,:]==0):
      front_B = data_norm.shape[1]- y

  for z in range(data_norm.shape[2]//2):
    if np.all(data_norm[:,:,z]==0):
      bottom_B = z

  for z in range(data_norm.shape[2]//2):
    if np.all(data_norm[:,:,-z]==0):
      top_B = data_norm.shape[2]-z

  return left_B,right_B,back_B,front_B,bottom_B,top_B

def Preprocess_Data():

    '''
    This function preprocesses the data, and saves them into an npz format, in the 1
    Prepro_Data folder. First we find the boundaries of the data, to find the largest
    brain. Then we pad with zeros all the other ones, up to that max size. Finally,
    we save the data, thus centering them.

    '''
    plt = platform.system()
    if plt == "Windows":
        slash = "\\"
    else:
        slash = "/"

    model_path = slash+"Models"+slash+"SonicNet"+slash
    config_name = "ConfigMRI_SonicNet.json"


    set_system_settings()
    base_path = os.path.dirname(os.path.realpath(__file__))



    main_path = base_path + slash + "Input_Data" + slash
    output_path = base_path + slash + "Prepro_Data" + slash
    subject_list = os.listdir(main_path)

    Max_diff_X = 0
    Max_diff_Y = 0
    Max_diff_Z = 0

    for subject in subject_list:
#if True:
        #aseg_path_specific = main_path + subject + "/aseg.mgz"
        norm_path_specific = main_path + subject 
        print("Processing labels from" + norm_path_specific)

        #nii_aseg = nb.load(aseg_path_specific)
        #data_aseg = np.asarray(nii_aseg.dataobj) #LABELS_Y

        nii_norm = nb.load(norm_path_specific)
        data_norm = np.asarray(nii_norm.dataobj) #DATA_X
        counter =0

        left_B,right_B,back_B,front_B,bottom_B,top_B = find_boundaries(data_norm)
        if (right_B - left_B)>Max_diff_X:
            Max_diff_X = right_B - left_B
        if (front_B - back_B)>Max_diff_Y:
            Max_diff_Y = front_B - back_B
        if (top_B - bottom_B)>Max_diff_Z:
            Max_diff_Z = top_B - bottom_B
    counter =0
    for subject in subject_list:
#if True:
        #aseg_path_specific = main_path + subject + "/aseg.mgz"
        norm_path_specific = main_path + subject 
        print("Processing labels from" + norm_path_specific)

        #nii_aseg = nb.load(aseg_path_specific)
        #data_aseg = np.asarray(nii_aseg.dataobj) #LABELS_Y

        nii_norm = nb.load(norm_path_specific)
        data_norm = np.asarray(nii_norm.dataobj) #DATA_X
        

        left_B,right_B,back_B,front_B,bottom_B,top_B = find_boundaries(data_norm)

        left_B,right_B,back_B,front_B,bottom_B,top_B = find_standard_boundaries(left_B,right_B,back_B,front_B,bottom_B,top_B,Max_diff_X,Max_diff_Y,Max_diff_Z)
        assert right_B - left_B==Max_diff_X
        assert front_B - back_B==Max_diff_Y
        assert top_B - bottom_B==Max_diff_Z
        data = data_norm[left_B:right_B, back_B:front_B, bottom_B:top_B]
        #labels_Y = data_aseg[left_B:right_B, back_B:front_B, bottom_B:top_B]
  

        data_X_normalized = func_nrm_range_zero_one(data, max)

        data_X = data_X_normalized
        Data_X_File = output_path + "subject_" + str(counter) + slash + "Data_X.npz"
        folder = output_path + "subject_" + str(counter)
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        #Data_Y_File = main_path + subject + "/Labels_Y.npz"
        np.savez_compressed(Data_X_File,data_X)
        #np.savez_compressed(Data_Y_File,labels_Y)
        counter+=1



if __name__ == '__main__':



    Preprocess_Data()

