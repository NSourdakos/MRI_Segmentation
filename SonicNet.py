import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, concatenate, BatchNormalization
from tensorflow.keras.layers import Conv3D, UpSampling3D, Conv3DTranspose, MaxPooling3D, Reshape
from tensorflow.keras.layers import add
from tensorflow.keras.layers import LeakyReLU, Reshape, Lambda, Softmax
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import numpy as np
import os
import gc
import platform
from Utils import *

def dice_coef(y_true, y_pred):
	'''
	This function, calculates the dice coefficient, for use in the loss function
	'''
	y_true_f = tf.squeeze(y_true)
	y_true_f = tf.cast(y_true_f, tf.float32)
	y_pred_f = tf.squeeze(y_pred)
	y_pred_f = tf.cast(y_pred_f, tf.float32)
	intersection = tf.math.reduce_sum(tf.math.multiply(y_true_f,y_pred_f))
	smooth = 0.0001
	return (2. * intersection + smooth) / (tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f) + smooth)

def mixed_dice_loss(y_true, y_pred):
	'''
	This function, calculates the loss function, both with the categorical
	crossentropy and the dice coefficient. Here a weight of 10 is given to the
	dice coefficient, to make it comparable with the loss from categorical crossentropy
	Also, the number of classes is defined here as well.
	'''

	cat_loss = tf.keras.losses.CategoricalCrossentropy()
	alpha = 10
	beta = 1
	dice=0
	
	num_classes = 30 #Need to pass that from somewhere
	#dice_weights = np.ones(num_classes)
	for index in range(num_classes):
		dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
	dice_loss = -dice/num_classes # taking average
	weighted_loss = beta*cat_loss(y_true, y_pred) + alpha*dice_loss
	return weighted_loss
	
def mixed_Weighted_dice_loss_DIFF(y_true, y_pred):
	cat_loss = tf.keras.losses.CategoricalCrossentropy()
	alpha = 5
	beta = 1
	dice=0
	num_classes = 30
	smooth = 0.0001

	#WEIGHTED DICE PART
	dice_weights = np.ones(num_classes)
	dice_weights[5] = 5
	dice_weights[12] = 5
	dice_weights[17] = 5
	dice_weights[18] = 5
	dice_weights[19] = 5
	dice_weights[21] = 5
	dice_weights = tf.cast(dice_weights, tf.float32)

	y_true_f = tf.reshape(y_true, [-1,num_classes])
	y_true_f = tf.cast(y_true_f, tf.float32)
	y_pred_f = tf.reshape(y_true, [-1,num_classes])
	y_pred_f = tf.cast(y_pred_f, tf.float32)
	intersection = tf.math.reduce_sum(tf.math.multiply(y_true_f,y_pred_f), axis = 0)
	dice = (2. * intersection + smooth) / (tf.math.reduce_sum(y_true_f, axis = 0) + tf.math.reduce_sum(y_pred_f, axis = 0) + smooth)
	dice_w = tf.math.multiply(dice_weights,dice)
	dice_loss = -tf.math.reduce_sum(dice_w)/num_classes # taking average
	weighted_loss = 8+beta*cat_loss(y_true, y_pred) + alpha*dice_loss
	return weighted_loss

def mixed__weighted_dice_loss(y_true, y_pred):
	'''
	A weighted version of "mixed_dice_loss", increasing the loss again for
	the less prevalent classes. Experiments did not show any general improvement.
	Might be useful to strengthen the predictions for a particular class though.
	'''
	cat_loss = tf.keras.losses.CategoricalCrossentropy()
	alpha = 10
	beta = 1
	dice=0
	dice_weight = 5
	num_classes = 30
	dice_weights = np.ones(num_classes)
	for i in range(num_classes):
		if i > 0:
			dice_weights[i] *= dice_weight 
	for i in range(num_classes):
		if i > 3:
			dice_weights[i] *= 3
	for index in range(num_classes):
		dice += dice_weights[index]*dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
	dice_loss = -dice/num_classes # taking average
	weighted_loss = 10+ beta*cat_loss(y_true, y_pred) + alpha*dice_loss
	return weighted_loss

def CustomConv3D(x_in, nf, strides=1, kernel_size = 3):
	"""
	Custom convolution module including convolution followed by batch normalization and leakyrelu
	"""
	x_out = Conv3D(nf, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
	#print("AAAAA", x_out.shape)
	x_out = BatchNormalization()(x_out)
	x_out = LeakyReLU(0.2)(x_out)
	return x_out

def CustomConv3DTranspose(x_in, nf, strides=2, kernel_size = 3):
	"""
	Custom convolution module including transpose convolution followed by batch normalization and leakyrelu
	"""
	x_out = Conv3DTranspose(nf, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
	#print("AAAAA", x_out.shape)
	x_out = BatchNormalization()(x_out)
	x_out = LeakyReLU(0.2)(x_out)
	return x_out

def Nonlocal3D(x_in, strides=1, kernel_size =1 ):
	'''
	Function that calculates the attention

	H,W,D are the 3 dimensions of the data cube
	phi theta and psi play the role of keys, queries and values found in traditional attention models.

	This layer is only called in the deeper parts of the Unet, as it unfolds the data cube, and on
	the early layers the key-query matrix becomes too much for the memory to handle.
	'''
	#nf_inter = int(x_in.shape[4]//2)
	nf_inter = int(x_in.shape[4])*4
	nf = int(x_in.shape[4])
	H = int(x_in.shape[1])
	W = int(x_in.shape[2])
	D = int(x_in.shape[3])
	nf_outer = int(nf_inter*2)
	theta = Conv3D(nf_inter, kernel_size=1, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
	theta = BatchNormalization()(theta)
	theta_reshaped = Reshape((H*W*D,nf_inter))(theta)

	phi = Conv3D(nf_inter, kernel_size=1, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
	phi = BatchNormalization()(phi)
	phi_reshaped = Reshape((H*W*D,nf_inter))(phi)

	phi_theta = tf.matmul(theta_reshaped, tf.transpose(phi_reshaped, [0, 2, 1]))
	#phi_theta = np.matmul(theta_reshaped, phi_reshaped)
	phi_theta_softmax = Softmax()(phi_theta)

	
	psi = Conv3D(nf_inter, kernel_size=1, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
	psi = BatchNormalization()(psi)
	psi_reshaped = Reshape((H*W*D,nf_inter))(psi)

	phi_psi_theta = tf.matmul(phi_theta_softmax, psi_reshaped)
	#phi_psi_theta = np.matmul(phi_theta_softmax, psi_reshaped)
	phi_psi_theta = Reshape((H,W,D,nf_inter))(phi_psi_theta)

	final = Conv3D(nf, kernel_size=1, padding='same',kernel_initializer='he_normal', strides=strides)(phi_psi_theta)
	final = BatchNormalization()(final)
	Zeta = tf.keras.layers.Add()([x_in, final])
	
	del phi_theta
	del phi_psi_theta
	gc.collect()

	return Zeta


class SonicNet:
	'''
	The class of the Network. It consists of 6 slim UNets, connected sequentially and densely.
	'''
	def __init__(self,config, load_checkpoint=None, print_model_summary=True):


		#super().__init__()
		self.config = config
		self.verbosity = config['training']['verbosity']

		

		self.optimizer = self.get_optimizer()
		#self.out_1_loss = self.get_out_1_loss()
		#self.out_2_loss = self.get_out_2_loss()
		#self.metrics = self.get_metrics()
		self.epoch_num = 0
		self.checkpoints_path = ''
		self.samples_path = ''
		self.history_filename = ''
		#self.input_size = input_size
		self.input_size =  self.config['model']['input_size']
		#self.num_classes = num_classes
		self.num_classes = self.config['model']['num_classes']
		


		self.model = self.setup_model(load_checkpoint, print_model_summary)

	def setup_model(self, load_checkpoint=None, print_model_summary=True):
		'''
		This is the function that sets up the model.
		It checks wether there is a pretrained model in the given directory.
		If not it creates it, else it just loads the stored weights of the latest epoch
		'''

		plt = platform.system()
		if plt == "Windows":
			slash = "\\"
		else:
			slash = "/"
		self.checkpoints_path = os.path.join(self.config['training']['path'], 'checkpoints')
		self.samples_path = os.path.join(self.config['training']['path'], 'samples')
		self.history_filename = 'history_' + self.config['training']['path'][
											 self.config['training']['path'].rindex(slash) + 1:] + '.csv'

		model = self.build_model()

		if os.path.exists(self.checkpoints_path) and dir_contains_files(self.checkpoints_path):

			if load_checkpoint is not None:
				last_checkpoint_path = load_checkpoint
				self.epoch_num = 0
			else:
				checkpoints = os.listdir(self.checkpoints_path)
				checkpoints.sort(key=lambda x: os.stat(os.path.join(self.checkpoints_path, x)).st_mtime)
				last_checkpoint = checkpoints[-1]
				last_checkpoint_path = os.path.join(self.checkpoints_path, last_checkpoint)
				self.epoch_num = int(last_checkpoint[11:16])
			print('Loading model from epoch: %d' % self.epoch_num)
			model.load_weights(last_checkpoint_path)

		else:
			print('Building new model...')

			if not os.path.exists(self.config['training']['path']):
				os.mkdir(self.config['training']['path'])

			if not os.path.exists(self.checkpoints_path):
				os.mkdir(self.checkpoints_path)

			self.epoch_num = 0

		if not os.path.exists(self.samples_path):
			os.mkdir(self.samples_path)

		if print_model_summary:
			model.summary()



		config_path = os.path.join(self.config['training']['path'], 'config.json')
		if not os.path.exists(config_path):
			pretty_json_dump(self.config, config_path)

		if print_model_summary:
			pretty_json_dump(self.config)
		return model
	
	def get_optimizer(self):
		'''
		Function that creates the optimizer. Here we are using Adam as a default
		Experimentation was also done with RMSprop, but no improvement was made

		'''
		return keras.optimizers.Adam(lr=self.config['optimizer']['lr'], decay=self.config['optimizer']['decay'],
									 epsilon=self.config['optimizer']['epsilon'])


	def get_callbacks(self):
		'''
		Function of callbacks. There is an adaptive learning rate on plateau, 
		however it has been effectively disabled from the configurations.
		Rest are for checkpoint creation, and the history file
		'''

		return [
			keras.callbacks.ReduceLROnPlateau(patience=self.config['training']['early_stopping_patience'] / 2,
											  cooldown=self.config['training']['early_stopping_patience'] / 4,
											  verbose=1),
			#keras.callbacks.EarlyStopping(patience=self.config['training']['early_stopping_patience'], verbose=1,
			#                              monitor='loss'),
			keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoints_path, 'checkpoint.{epoch:05d}-{val_loss:.3f}.hdf5')),
			keras.callbacks.CSVLogger(os.path.join(self.config['training']['path'], self.history_filename), append=True)
		]

	def fit_model(self, train_set_generator, num_train_samples, test_set_generator, num_test_samples, num_epochs):

		print('Fitting model with %d training samples and %d test samples...' % (num_train_samples, num_test_samples))

		self.model.fit_generator(train_set_generator,
								 num_train_samples,
								 epochs=num_epochs,
								 validation_data=test_set_generator,
								 #nb_val_samples=num_test_samples,
								 callbacks=self.get_callbacks(),
								 verbose=self.verbosity,
								 initial_epoch=self.epoch_num)


	def predict(self, inputs):
	  return self.model.predict_on_batch(inputs)

	def get_metrics(self):

		return [
			keras.metrics.mean_absolute_error,
			self.valid_mean_absolute_error
		]

	def valid_mean_absolute_error(self, y_true, y_pred):
		return keras.backend.mean(
			keras.backend.abs(y_true[:, 1:-2] - y_pred[:, 1:-2]))


	
	def build_model(self):

		filter_number = 64
		filter_number2 = 128
		final_filters = 128

		inputs = Input(self.input_size, name='data_input')
		conv1 = CustomConv3D((inputs),filter_number)

		#First Loop
		conv1_bottom = CustomConv3D((conv1),filter_number)
		pool1_bottom = MaxPooling3D(pool_size=(2, 2, 2))(conv1_bottom)
		
		conv1_right_low = CustomConv3D((pool1_bottom),filter_number)
		conv1_right_low = CustomConv3D((conv1_right_low),filter_number)
		pool1_right_low = MaxPooling3D(pool_size=(2, 2, 2))(conv1_right_low)

		conv1_right_high = CustomConv3D((pool1_right_low),filter_number)
		conv1_right_high = CustomConv3D((conv1_right_high),filter_number)
		pool1_right_high = MaxPooling3D(pool_size=(2, 2, 2))(conv1_right_high)
	
		conv1_top = CustomConv3D((pool1_right_high),filter_number)
		conv1_top = CustomConv3D((conv1_top),filter_number)
		#up1_top = UpSampling3D(size = (2,2,2))(conv1_top)
		up1_top = CustomConv3DTranspose((conv1_top),filter_number)
	
		merge1_left_high = concatenate([up1_top,conv1_right_high], axis = 4)
		conv1_left_high = CustomConv3D((merge1_left_high),filter_number)
		conv1_left_high = CustomConv3D((conv1_left_high),filter_number)
		#up1_left_high = UpSampling3D(size = (2,2,2))(conv1_left_high)
		up1_left_high = CustomConv3DTranspose((conv1_left_high),filter_number)

		merge1_left_low = concatenate([up1_left_high,conv1_right_low], axis = 4)
		conv1_left_low = CustomConv3D((merge1_left_low),filter_number)
		conv1_left_low = CustomConv3D((conv1_left_low),filter_number)
		#up1_left_low = UpSampling3D(size = (2,2,2))(conv1_left_low)
		up1_left_low = CustomConv3DTranspose((conv1_left_low),filter_number)
		
		merge1_full = concatenate([up1_left_low,conv1_bottom], axis = 4)
		conv1_full = CustomConv3D((merge1_full),filter_number)

		#Second Loop

		conv2 = CustomConv3D((inputs),filter_number)
		merge2_start = concatenate([conv1_full,conv2], axis = 4)

		conv2_bottom = CustomConv3D((merge2_start),filter_number)
		pool2_bottom = MaxPooling3D(pool_size=(2, 2, 2))(conv2_bottom)
		
		conv2_right_low = CustomConv3D((pool2_bottom),filter_number)
		dense2_right_low = concatenate([conv2_right_low,conv1_right_low,conv1_left_low], axis = 4)
		conv2_right_low = CustomConv3D((dense2_right_low),filter_number)
		pool2_right_low = MaxPooling3D(pool_size=(2, 2, 2))(conv2_right_low)

		conv2_right_high = CustomConv3D((pool2_right_low),filter_number)
		dense2_right_high = concatenate([conv2_right_high,conv1_right_high,conv1_left_high], axis = 4)
		conv2_right_high = CustomConv3D((dense2_right_high),filter_number)
		pool2_right_high = MaxPooling3D(pool_size=(2, 2, 2))(conv2_right_high)

		
	
		conv2_top = CustomConv3D((pool2_right_high),filter_number)
		dense2_top = concatenate([conv2_top,conv1_top], axis = 4)
		conv2_top = CustomConv3D((dense2_top),filter_number)
		#up2_top = UpSampling3D(size = (2,2,2))(conv2_top)
		up2_top = CustomConv3DTranspose((conv2_top),filter_number)
	
		merge2_left_high = concatenate([up2_top,conv2_right_high], axis = 4)
		conv2_left_high = CustomConv3D((merge2_left_high),filter_number)
		dense2_left_high = concatenate([conv2_left_high,conv1_left_high,conv1_right_high], axis = 4)
		conv2_left_high = CustomConv3D((dense2_left_high),filter_number)
		#up2_left_high = UpSampling3D(size = (2,2,2))(conv2_left_high)
		up2_left_high = CustomConv3DTranspose((conv2_left_high),filter_number)

		merge2_left_low = concatenate([up2_left_high,conv2_right_low], axis = 4)
		conv2_left_low = CustomConv3D((merge2_left_low),filter_number)
		dense2_left_low = concatenate([conv2_left_low,conv1_left_low,conv1_right_low], axis = 4)
		conv2_left_low = CustomConv3D((dense2_left_low),filter_number)
		#up2_left_low = UpSampling3D(size = (2,2,2))(conv2_left_low)
		up2_left_low = CustomConv3DTranspose((conv2_left_low),filter_number)


		merge2_full = concatenate([up2_left_low,conv2_bottom], axis = 4)
		conv2_full = CustomConv3D((merge2_full),filter_number)


		#Third Loop

		conv3 = CustomConv3D((inputs),filter_number)
		#merge2_start = concatenate([conv1_full,conv2], axis = 4)

		merge3_bottom = concatenate([conv3,conv2_full,conv1_full], axis = 4)
		conv3_bottom = CustomConv3D((merge3_bottom),filter_number)
		pool3_bottom = MaxPooling3D(pool_size=(2, 2, 2))(conv3_bottom)
		
		conv3_right_low = CustomConv3D((pool3_bottom),filter_number)
		dense3_right_low = concatenate([conv3_right_low,conv2_right_low,conv1_right_low,conv2_left_low,conv1_left_low], axis = 4)
		conv3_right_low = CustomConv3D((dense3_right_low),filter_number)
		pool3_right_low = MaxPooling3D(pool_size=(2, 2, 2))(conv3_right_low)

		conv3_right_high = CustomConv3D((pool3_right_low),filter_number)
		dense3_right_high = concatenate([conv3_right_high,conv2_right_high,conv1_right_high,conv2_left_high,conv1_left_high], axis = 4)
		conv3_right_high = CustomConv3D((dense3_right_high),filter_number)
		pool3_right_high = MaxPooling3D(pool_size=(2, 2, 2))(conv3_right_high)

		
	
		conv3_top = CustomConv3D((pool3_right_high),filter_number)
		dense3_top = concatenate([conv3_top,conv2_top,conv1_top], axis = 4)
		conv3_top = CustomConv3D((dense3_top),filter_number)
		#up3_top = UpSampling3D(size = (2,2,2))(conv3_top)
		up3_top = CustomConv3DTranspose((conv3_top),filter_number)
	
		merge3_left_high = concatenate([up3_top,conv3_right_high], axis = 4)
		conv3_left_high = CustomConv3D((merge3_left_high),filter_number)
		dense3_left_high = concatenate([conv3_left_high,conv2_left_high,conv1_left_high,conv2_right_high,conv1_right_high], axis = 4)
		conv3_left_high = CustomConv3D((dense3_left_high),filter_number)
		#up3_left_high = UpSampling3D(size = (2,2,2))(conv3_left_high)
		up3_left_high = CustomConv3DTranspose((conv3_left_high),filter_number)

		merge3_left_low = concatenate([up3_left_high,conv3_right_low], axis = 4)
		conv3_left_low = CustomConv3D((merge3_left_low),filter_number)
		dense3_left_low = concatenate([conv3_left_low,conv2_left_low,conv1_left_low,conv2_right_low,conv1_right_low], axis = 4)
		conv3_left_low = CustomConv3D((dense3_left_low),filter_number)
		#up3_left_low = UpSampling3D(size = (2,2,2))(conv3_left_low)
		up3_left_low = CustomConv3DTranspose((conv3_left_low),filter_number)


		merge3_full = concatenate([up3_left_low,conv3_bottom], axis = 4)
		conv3_full = CustomConv3D((merge3_full),filter_number)


		#Fourth Loop

		conv4 = CustomConv3D((inputs),filter_number)
		#merge2_start = concatenate([conv1_full,conv2], axis = 4)

		merge4_bottom = concatenate([conv4,conv3_full,conv2_full,conv1_full], axis = 4)
		conv4_bottom = CustomConv3D((merge4_bottom),filter_number)
		pool4_bottom = MaxPooling3D(pool_size=(2, 2, 2))(conv4_bottom)
		
		conv4_right_low = CustomConv3D((pool4_bottom),filter_number)
		dense4_right_low = concatenate([conv4_right_low,conv3_right_low,conv2_right_low,conv1_right_low,conv3_left_low,conv2_left_low,conv1_left_low], axis = 4)
		conv4_right_low = CustomConv3D((dense4_right_low),filter_number)
		pool4_right_low = MaxPooling3D(pool_size=(2, 2, 2))(conv4_right_low)

		conv4_right_high = CustomConv3D((pool4_right_low),filter_number)
		dense4_right_high = concatenate([conv4_right_high,conv3_right_high,conv2_right_high,conv1_right_high,conv3_left_high,conv2_left_high,conv1_left_high], axis = 4)
		conv4_right_high = CustomConv3D((dense4_right_high),filter_number)
		pool4_right_high = MaxPooling3D(pool_size=(2, 2, 2))(conv4_right_high)

		
	
		conv4_top = CustomConv3D((pool4_right_high),filter_number)
		dense4_top = concatenate([conv4_top,conv3_top,conv2_top,conv1_top], axis = 4)
		conv4_top = CustomConv3D((dense4_top),filter_number)
		#up4_top = UpSampling3D(size = (2,2,2))(conv4_top)
		up4_top = CustomConv3DTranspose((conv4_top),filter_number)
	
		merge4_left_high = concatenate([up4_top,conv4_right_high], axis = 4)
		conv4_left_high = CustomConv3D((merge4_left_high),filter_number)
		dense4_left_high = concatenate([conv4_left_high,conv3_left_high,conv2_left_high,conv1_left_high,conv3_right_high,conv2_right_high,conv1_right_high], axis = 4)
		conv4_left_high = CustomConv3D((dense4_left_high),filter_number)
		#up4_left_high = UpSampling3D(size = (2,2,2))(conv4_left_high)
		up4_left_high = CustomConv3DTranspose((conv4_left_high),filter_number)


		merge4_left_low = concatenate([up4_left_high,conv4_right_low], axis = 4)
		conv4_left_low = CustomConv3D((merge4_left_low),filter_number)
		dense4_left_low = concatenate([conv4_left_low,conv3_left_low,conv2_left_low,conv1_left_low,conv3_right_low,conv2_right_low,conv1_right_low], axis = 4)
		conv4_left_low = CustomConv3D((dense4_left_low),filter_number)
		#up4_left_low = UpSampling3D(size = (2,2,2))(conv4_left_low)
		up4_left_low = CustomConv3DTranspose((conv4_left_low),filter_number)


		merge4_full = concatenate([up4_left_low,conv4_bottom], axis = 4)
		conv4_full = CustomConv3D((merge4_full),filter_number)


		#Deeper architecture-two more layers

		#Fifth Loop
		conv5 = CustomConv3D((inputs),filter_number2)
		merge5_bottom = concatenate([conv5,conv4_full,conv3_full,conv2_full,conv1_full], axis = 4)
		conv5_bottom = CustomConv3D((merge5_bottom),filter_number2)
		pool5_bottom = MaxPooling3D(pool_size=(2, 2, 2))(conv5_bottom)
		
		conv5_right_low = CustomConv3D((pool5_bottom),filter_number2)
		dense5_right_low = concatenate([conv5_right_low,conv4_right_low,conv3_right_low,conv2_right_low,conv1_right_low,conv4_left_low,conv3_left_low,conv2_left_low,conv1_left_low], axis = 4)
		conv5_right_low = CustomConv3D((dense5_right_low),filter_number2)
		pool5_right_low = MaxPooling3D(pool_size=(2, 2, 2))(conv5_right_low)

		conv5_right_high = CustomConv3D((pool5_right_low),filter_number2)
		dense5_right_high = concatenate([conv5_right_high,conv4_right_high,conv3_right_high,conv2_right_high,conv1_right_high,
										 conv4_left_high,conv3_left_high,conv2_left_high,conv1_left_high], axis = 4)
		conv5_right_high = CustomConv3D((dense5_right_high),filter_number2)
		pool5_right_high = MaxPooling3D(pool_size=(2, 2, 2))(conv5_right_high)

		
	
		conv5_top = CustomConv3D((pool5_right_high),filter_number2)
		dense5_top = concatenate([conv5_top,conv4_top,conv3_top,conv2_top,conv1_top], axis = 4)
		conv5_top = CustomConv3D((dense5_top),filter_number2)
		#up5_top = UpSampling3D(size = (2,2,2))(conv5_top)
		up5_top = CustomConv3DTranspose((conv5_top),filter_number2)

	
		merge5_left_high = concatenate([up5_top,conv5_right_high], axis = 4)
		conv5_left_high = CustomConv3D((merge5_left_high),filter_number2)
		dense5_left_high = concatenate([conv5_left_high,conv4_left_high,conv3_left_high,conv2_left_high,conv1_left_high,
										conv4_right_high,conv3_right_high,conv2_right_high,conv1_right_high], axis = 4)
		conv5_left_high = CustomConv3D((dense5_left_high),filter_number2)
		#up5_left_high = UpSampling3D(size = (2,2,2))(conv5_left_high)
		up5_left_high = CustomConv3DTranspose((conv5_left_high),filter_number2)

		merge5_left_low = concatenate([up5_left_high,conv5_right_low], axis = 4)
		conv5_left_low = CustomConv3D((merge5_left_low),filter_number2)
		dense5_left_low = concatenate([conv5_left_low,conv4_left_low,conv3_left_low,conv2_left_low,conv1_left_low,
									   conv4_right_low,conv3_right_low,conv2_right_low,conv1_right_low], axis = 4)
		conv5_left_low = CustomConv3D((dense5_left_low),filter_number2)
		#up5_left_low = UpSampling3D(size = (2,2,2))(conv5_left_low)
		up5_left_low = CustomConv3DTranspose((conv5_left_low),filter_number2)


		merge5_full = concatenate([up5_left_low,conv5_bottom], axis = 4)
		conv5_full = CustomConv3D((merge5_full),filter_number2)

		#Sixth Loop
		merge6_bottom = concatenate([conv5_full,conv4_full,conv3_full,conv2_full,conv1_full], axis = 4)
		conv6_bottom = CustomConv3D((merge6_bottom),filter_number2)
		pool6_bottom = MaxPooling3D(pool_size=(2, 2, 2))(conv6_bottom)
		
		conv6_right_low = CustomConv3D((pool6_bottom),filter_number2)
		dense6_right_low = concatenate([conv6_right_low,conv5_right_low,conv4_right_low,conv3_right_low,conv2_right_low,conv1_right_low,
										conv5_left_low,conv4_left_low,conv3_left_low,conv2_left_low,conv1_left_low], axis = 4)
		conv6_right_low = CustomConv3D((dense6_right_low),filter_number2)
		pool6_right_low = MaxPooling3D(pool_size=(2, 2, 2))(conv6_right_low)

		conv6_right_high = CustomConv3D((pool6_right_low),filter_number2)
		dense6_right_high = concatenate([conv6_right_high,conv5_right_high,conv4_right_high,conv3_right_high,conv2_right_high,conv1_right_high,
										 conv5_left_high,conv4_left_high,conv3_left_high,conv2_left_high,conv1_left_high], axis = 4)
		conv6_right_high = CustomConv3D((dense6_right_high),filter_number2)
		pool6_right_high = MaxPooling3D(pool_size=(2, 2, 2))(conv6_right_high)

		
	
		conv6_top = CustomConv3D((pool6_right_high),filter_number2)
		dense6_top = concatenate([conv6_top,conv5_top,conv4_top,conv3_top,conv2_top,conv1_top], axis = 4)
		conv6_top = CustomConv3D((dense6_top),filter_number2)
		#up6_top = UpSampling3D(size = (2,2,2))(conv6_top)
		up6_top = CustomConv3DTranspose((conv6_top),filter_number2)
	
		merge6_left_high = concatenate([up6_top,conv6_right_high], axis = 4)
		conv6_left_high = CustomConv3D((merge6_left_high),filter_number2)
		dense6_left_high = concatenate([conv6_left_high,conv5_left_high,conv4_left_high,conv3_left_high,conv2_left_high,conv1_left_high,
										conv5_right_high,conv4_right_high,conv3_right_high,conv2_right_high,conv1_right_high], axis = 4)
		conv6_left_high = CustomConv3D((dense6_left_high),filter_number2)
		#up6_left_high = UpSampling3D(size = (2,2,2))(conv6_left_high)
		up6_left_high = CustomConv3DTranspose((conv6_left_high),filter_number2)

		merge6_left_low = concatenate([up6_left_high,conv6_right_low], axis = 4)
		conv6_left_low = CustomConv3D((merge6_left_low),filter_number2)
		dense6_left_low = concatenate([conv6_left_low,conv5_left_low,conv4_left_low,conv3_left_low,conv2_left_low,conv1_left_low,
									   conv5_right_low,conv4_right_low,conv3_right_low,conv2_right_low,conv1_right_low], axis = 4)
		conv6_left_low = CustomConv3D((dense6_left_low),filter_number2)
		#up6_left_low = UpSampling3D(size = (2,2,2))(conv6_left_low)
		up6_left_low = CustomConv3DTranspose((conv6_left_low),filter_number2)


		merge6_full = concatenate([up6_left_low,conv6_bottom], axis = 4)
		conv6_full = CustomConv3D((merge6_full),filter_number2)



		#Final Layer

		conv_final = CustomConv3D((conv6_full),final_filters)

		conv_final_2 = CustomConv3D((conv_final),final_filters)
		#merge_final = concatenate([conv_final,conv_final_2], axis = 4)
		#conv_final_3 = CustomConv3D((conv_final_2),final_filters)
		#conv9 = CustomConv3D((conv9),64)
		output = Conv3D(self.num_classes, 1, activation = 'sigmoid', name='data_output')(conv_final_2)
	
		output = Softmax()(output)

		model = Model(inputs = inputs, outputs = output)

		#model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		
		#model.compile(optimizer = self.get_optimizer(), loss = 'binary_crossentropy', metrics = ['accuracy'])
		#model.compile(optimizer = self.get_optimizer(), loss = "categorical_crossentropy", metrics = ['accuracy'])
		
		#model.compile(optimizer = self.get_optimizer(), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
		model.compile(optimizer = self.get_optimizer(), loss = mixed_Weighted_dice_loss_DIFF, metrics = ['accuracy'])
		#model.compile(optimizer = self.get_optimizer(), loss = mixed_weighted_dice_loss, metrics = ['accuracy'])

		model.summary()

	#if(pretrained_weights):
	#	model.load_weights(pretrained_weights)

		return model




	
   




	
   
