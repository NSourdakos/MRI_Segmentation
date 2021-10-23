Steps to use. This is mainly to use the pretrained network SonicNet, with a new dataset.

1 Unzip
2 Copy the data you want to use inside the folder "Input_Data". It is important that the datafiles are in a format that is
  readable by nibabel. Also, the data files should not be inside other folders, otherwise the preprocessing step will fail.
3 Run Preprocess_Data. Your folder "Prepro_Data" should now be full witht he centered data, in npz format
4 Run the  Predict_SonicNet.py to create output accuracies and output labeled data in the "Output_Data" folder

Dependencies
tensorflow 2.4.0 gpu
nibabel


Config


 "dataset": {
        
        "in_memory_percentage": 1,  	#Not used
        "overlap_percentage" : 0.5,		#The overlap percentage between successive data cubes
        "shape": [128, 128, 160],		#The shape of the data
        "num_condition_classes": 30,	#The number of classes
        "path": "G:\\Books\\AI\\CODE\\Internship\\Results\\",	#None of the paths here are used, the pathis taken from where the file is
        "dictionary_path": "G:\\Books\\AI\\CODE\\Internship\\label_convert_final.npy",
        "test_data":100					#The number of test data-also the number excluded from training

        
    },
    "model": {
        "input_size": [48,48,48,1] ,	#The input cube size
        "num_classes": 30,				#Number of classes-fixed
        "U-depth": 5,					#not used
        "Dropout": 0.3,					#also not used, did not improve anything
        "filters": {
            "widths":[32,48,64,96,128,192,256,384]	#a list of the number of filters used 
        },
        "num_stacks": 2,				#not used
        "deep_supervision": 0			#also not used for now, did not improve anything
    },
    "optimizer": {						#optimizer is fixed to adam
        "decay": 0.0,
        "epsilon": 1e-08,
        "lr": 0.00005,
        "momentum": 0.9,
        "type": "RMSprop"
    },
    "training": {
        "batch_size": 10,			
        "early_stopping_patience": 100,
        "num_epochs": 100,				
        "num_test_samples": 441,	#these parameters are from when the numbber of samples was hardcoded, now it is calculated from the overlap
        "num_train_samples": 441,
        "path": "G:\\Books\\AI\\CODE\\Internship\\Models\\UNET_PlusPlus\\Training\\",
        "verbosity": 2
    }