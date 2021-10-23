import os
import numpy as np
import logging
import gc
from tensorflow import keras

import matplotlib.pyplot as plt



class FreeSurfer_Dataset():
    '''
    This class contains the dataset and the generator for the data for processing.
    Since the whole dataset is too much for the memory to handle, each epoch
    data from one patient is loaded through the data generator. All the important
    parameters are loaded through the config file.

    '''
    def __init__(self, config, model):

        self.model = model
        self.path = config['dataset']['path']
        self.batch_size = config['training']['batch_size']
        self.in_memory_percentage = config['dataset']['in_memory_percentage']
        self.config = config
        self.data_cube_size = config['model']["input_size"][0]
        self.data_overlap_percentage = config['dataset']['overlap_percentage']
 
        self.file_dir_path = os.path.dirname(os.path.realpath(__file__))
        #self.base_dir = "/content/drive/My Drive/Internship/data/OASIS1/Results/"
        self.base_dir = self.file_dir_path + "\\Prepro_Data\\"
        self.data_Counter_X = 0
        self.data_Counter_Y = 0
        self.data_Counter_Z = 0
    def createDataGenerator(self, set):
        return FreeSurfer_Dataset.DataGenerator(self, set)
    class DataGenerator(keras.utils.Sequence):
        '''


        This is the data generator class
        Data generation is called on initiallization and on epoch end
        '''
        def __init__(self, datasetInstance, set):
            '''


            Because the whole brain cannot fit into the memory, chunks of 64x64x64
            are loaded sequentially. How many chunks depends on the parameter called
            data_overlap_percentage. We have then a sequential loading of the data
            
            data_Counter_X, data_Counter_Y, data_Counter_Z are the counters that
            determine the position of the current cube of data, compared to the
            whole brain. Seeing their max values also informs the decision of the parameter
            of loaded datapoins(cubes). For example, with 80% overlap, and 64 size cubes
            we get a total of 7x7x9=441 datapoints.




            '''
            self.datasetInstance = datasetInstance

            self.new_Subject = 0
            self.data_Counter = 0
            self.data_Enh_Flag = 0

            self.data_Counter_X = 0
            self.data_Counter_Y = 0
            self.data_Counter_Z = 0

            self.on_epoch_end()
            self.file_dir_path = os.path.dirname(os.path.realpath(__file__))
            #self.base_dir = "/content/drive/My Drive/Internship/data/OASIS1/Results/"
            self.base_dir = self.file_dir_path + "\\Prepro_Data\\"
            #self.base_dir = "/content/drive/My Drive/Internship/data/OASIS1/Results/"
            self.subject_list = os.listdir(self.base_dir)
            self.subject_list.sort()
            self.left_out_data = 40
            rand = np.random.randint(1, len(self.subject_list) -self.left_out_data )
            self.subj_name = self.subject_list[rand]
            #i= np.random.randint(1, 340)
        # Generate data
            #print("data_init",i)
            try:
              data_X = np.load(f'{self.base_dir+self.subj_name}/Data_X_Centered_Trimmed.npz',allow_pickle=True)
              print(self.base_dir+self.subj_name)
              self.data_X = data_X['arr_0']
              data_X.close()
              del data_X

              labels_Y = np.load(f'{self.base_dir+self.subj_name}/Labels_Y_Centered_Trimmed.npz',allow_pickle=True)
              self.labels_Y = labels_Y['arr_0']
              labels_Y.close()
              del labels_Y
            except FileNotFoundError:
              print('Ooops, file not found')
              print(self.base_dir+self.subj_name)
            

        def __len__(self):
        
            return self.datasetInstance.config['training']['num_train_samples']

        def __getitem__(self, index):
        
        # Generate indexes of the batch

        # Generate data
            batch = self.__data_generation()

            return batch

        def on_epoch_end(self):
            '''



            Method that bring the new person data when the training epoch ends

            '''

            self.file_dir_path = os.path.dirname(os.path.realpath(__file__))
            #self.base_dir = "/content/drive/My Drive/Internship/data/OASIS1/Results/"
            self.base_dir = self.file_dir_path + "\\Prepro_Data\\"
            
            self.subject_list = os.listdir(self.base_dir)
            self.subject_list.sort()
            self.left_out_data = 40
            rand = np.random.randint(1, len(self.subject_list) -self.left_out_data )
            self.subj_name = self.subject_list[rand]
            self.new_Subject = 0
            self.data_Enh_Flag = 0
            self.data_Counter = 0
            #self.data_Enh_Counter = 0
            self.data_Counter_X = 0
            self.data_Counter_Y = 0
            self.data_Counter_Z = 0
 
        # Generate data
            
            try:
              data_X = np.load(f'{self.base_dir+self.subj_name}/Data_X_Centered_Trimmed.npz',allow_pickle=True)
              self.data_X = data_X['arr_0']
              print(type(self.data_X))
              data_X.close()
              del data_X
              print(self.base_dir+self.subj_name)
              labels_Y = np.load(f'{self.base_dir+self.subj_name}/Labels_Y_Centered_Trimmed.npz',allow_pickle=True)
              self.labels_Y = labels_Y['arr_0']
              labels_Y.close()
              del labels_Y
            except FileNotFoundError:
              print('Ooops, file not found')
            gc.collect()
            r=1

        def __data_generation(self):
            '''
            This is the core method that loads the 64x64x64 datapoints

            Calls the Calculate_Input_Output to update the counters, and create
            the data input-output pair, dependent of those counters
            '''
            batch_inputs = []
            batch_outputs = []

            if True: 
                
                
                
                       
                #print("data_XYZ",self.data_Counter_X, self.data_Counter_Y, self.data_Counter_Z)

                

    

                data_cube_size = self.datasetInstance.data_cube_size
                data_overlap_percentage = self.datasetInstance.data_overlap_percentage
                

                inputs,labels,self.data_Counter_X,self.data_Counter_Y,self.data_Counter_Z= self.datasetInstance.Calculate_Input_Output(self.data_X, self.labels_Y,
                                                                                          data_cube_size,data_overlap_percentage,self.data_Counter_X, self.data_Counter_Y, self.data_Counter_Z )




                correct_output = self.datasetInstance.Correct_labels(labels)


                batch_inputs.append(inputs)
                batch_outputs.append(correct_output)

                batch_inputs = np.asarray(batch_inputs, dtype='float32')

                batch_outputs = np.asarray(batch_outputs, dtype='uint8')

                gc.collect()

            return batch_inputs, batch_outputs

    def Calculate_Input_Output(self, data_X, labels_Y,data_cube_size,data_overlap_percentage, Counter_X, Counter_Y, Counter_Z):
      '''
      This function calculates the input and output cubes for the data generator.

      data_new_size is the "advancement" of the data cube each step. For instance
                    with a 64 cube and a 0.5 overlap, we have a data_new_size of 32.

      Counter_X_Limit calculates the number of cubes loaded, along direction X.
                      Same with Y and Z

      not_integer_X calculates whether, with the current data overlap, we have
                    an integer value of steps. default is 0, which means it IS and integer. If 
                    there is indeed a non integer value of steps, we round it up, by 
                    essentially increasing the limit by one, and taking the last cube from
                    the other end. For example, lets say we have a hube of 4, a new_size of 2, and 
                    the new limit of X is 8(so 9 in total, from 0 to 8). Then we will load , 
                    [0,3], [2,5], [4,7], and [5,8].
                    
      right_B, left_B etc, are the actual indices that indicate where we get our cubes from the whole

      '''
      data_new_size = int(data_cube_size*(1-data_overlap_percentage))
      Counter_X_Limit = int((data_X.shape[0] - data_cube_size)//data_new_size)
      Counter_Y_Limit = int((data_X.shape[1] - data_cube_size)//data_new_size)
      Counter_Z_Limit = int((data_X.shape[2] - data_cube_size)//data_new_size)
      not_integer_X=0
      not_integer_Y=0
      not_integer_Z=0

      #Case for not Integer
      if data_X.shape[0]> Counter_X_Limit*data_new_size + data_cube_size:
        #print("NOT INTEGER X")
        not_integer_X=1
        Counter_X_Limit +=1
      if data_X.shape[1]> Counter_Y_Limit*data_new_size + data_cube_size:
        #print("NOT INTEGER Y")
        not_integer_Y=1
        Counter_Y_Limit +=1
      if data_X.shape[2]> Counter_Z_Limit*data_new_size + data_cube_size:
        #print("NOT INTEGER Z")
        not_integer_Z=1
        Counter_Z_Limit +=1
      
      

      if not_integer_X==0:
        left_B= Counter_X*data_new_size
      else:
        if Counter_X==Counter_X_Limit:
          
          left_B = data_X.shape[0] - data_cube_size
        else:
          left_B= Counter_X*data_new_size

      if not_integer_Y==0:
        back_B= Counter_Y*data_new_size
      else:
        if Counter_Y==Counter_Y_Limit:
          
          back_B = data_X.shape[1] - data_cube_size
        else:
          back_B= Counter_Y*data_new_size

      if not_integer_Z==0:
        bottom_B= Counter_Z*data_new_size
      else:
        if Counter_Z==Counter_Z_Limit:
          
          bottom_B = data_X.shape[2] - data_cube_size
        else:
          bottom_B= Counter_Z*data_new_size


      right_B = left_B + data_cube_size
      front_B = back_B + data_cube_size
      top_B = bottom_B + data_cube_size

      inputs = data_X[left_B:right_B,back_B:front_B,bottom_B:top_B]
      labels=labels_Y[left_B:right_B,back_B:front_B,bottom_B:top_B]

      #print(left_B,"to",right_B)
      #print(back_B,"to",front_B)
      #print(bottom_B,"to",top_B)
      new_subject__,Counter_X, Counter_Y, Counter_Z = self.Increase_Counters(Counter_X, Counter_Y, Counter_Z,Counter_X_Limit, Counter_Y_Limit, Counter_Z_Limit)

      return inputs,labels,Counter_X, Counter_Y, Counter_Z



    def Increase_Counters(self,Counter_X,Counter_Y,Counter_Z, Counter_X_Limit, Counter_Y_Limit, Counter_Z_Limit):
      '''
      This method, given the current XYZ counters, and their limits calculated in Calculate_Input_Output
      increases them, first on the Z direction, then Y, and then X

      '''
      new_Subject = 0
      if Counter_X<0 or Counter_Y<0 or Counter_Z<0:
        print("Error, wrong XYZ counters")
      if Counter_X<=Counter_X_Limit:
        if Counter_Y<=Counter_Y_Limit:
          if Counter_Z<=Counter_Z_Limit-1:
            Counter_Z += 1
          else:
            Counter_Z = 0
            Counter_Y += 1
            if Counter_Y>Counter_Y_Limit:
              Counter_Y = 0
              Counter_Z = 0
              Counter_X += 1
              if Counter_X>Counter_X_Limit:
                Counter_Z = 0
                Counter_Y = 0
                Counter_X = 0
                new_Subject = 1
        else:
          Counter_Z = 0
          Counter_Y = 0  
          Counter_X += 1
          if X>Counter_X_Limit:
            Counter_Z = 0
            Counter_Y = 0
            Counter_X = 0 
            new_Subject = 1
      else:
        Counter_Z = 0
        Counter_Y = 0
        Counter_X = 0
        new_Subject = 1

      return new_Subject,Counter_X,Counter_Y,Counter_Z

    def Correct_labels(self,labels):

      '''
      This method converts the Freesurfer labels to our own label list, of 30
      different labels, that don't separate left from right, in an one-hot form.
      The label_conv_matrix has been created elsewhere and is being loaded.
      '''
      self.label_path = self.config['dataset']['dictionary_path']
      label_conv_matrix = np.load(self.label_path)
      label_dict = dict(zip(label_conv_matrix[:,0], label_conv_matrix[:,1]))
      #print(label_dict)
      label_num = (np.unique(label_conv_matrix[:,1])).shape[0]

      onehot_label_matrix = np.zeros((labels.shape[0],labels.shape[1],labels.shape[2],label_num))
      onehot_label_matrix = onehot_label_matrix.astype('uint8')
      for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
          for z in range(labels.shape[2]):
              categ = int(labels[x,y,z])
              if (categ in label_dict.keys()):
                new_categ=label_dict[categ]
                onehot_label_matrix[x,y,z,new_categ] = 1


      return onehot_label_matrix

    def load_dataset_2(self):
        '''
        This method creates the dataset instance on the training code

        '''

        self.file_dir_path = os.path.dirname(os.path.realpath(__file__))
        #self.base_dir = "/content/drive/My Drive/Internship/data/OASIS1/Results/"
        self.base_dir = self.file_dir_path + "\\Prepro_Data\\"

        #self.base_dir = "/content/drive/My Drive/Internship/data/OASIS1/Results/"
        self.subject_list = os.listdir(self.base_dir)

        rand = np.random.randint(0, len(self.subject_list) -40 )
        self.subj_name = self.subject_list[rand]

      
        return self


