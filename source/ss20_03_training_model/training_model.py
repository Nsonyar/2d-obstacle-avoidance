import h5py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from source.files.file_administration_v_02 import FileAdministration
tf.config.experimental \
.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# ******************************************************************************
# *************************** Custom Error class *******************************
# ******************************************************************************

class CustomError(Exception):
	''' Custom Class to provide the user with a proper error message in case
		some exception is raised.
	
	Args:
		msg: 	optional parameter to provide a message to the user
		idx: 	optional parameter to indicate further index information about
				specific files corresponding to the error raised
				
	Return: 	Error message
	'''
	def __init__(self, msg = None, idx = None):
		if msg:
			self.message = msg
			self.idx = idx
		else:
			self.message = None

	def __str__(self):
		if self.message and not(self.idx):
			return 'CustomError, {0}'.format(self.message)
		elif self.message and self.idx:
			return 'CustomError, {0}{1} !'.format(self.message,self.idx)
		else:
			return 'CustomError has been raised'

# ******************************************************************************
# ******************************* Main Class ***********************************
# ******************************************************************************

class Train:
	''' This script is responsible for training a Convolutional Neural Network
		given a dataset as input from Feature Extraction. Some concepts and some
		script snippets, were taken from [NGCG+19] as further described in the 
		main paper. To enjoy GPU support, the script should run from within the 
		provided docker container. It is suggested to run the script from the
		pipeline launcher, which incorporates it properly within the project.
		The different training parameters can be set manually as default values,
		or given through the command line in the pipeline launcher. The script
		consists of the architecture, parameter setup and a custom error class
		which prints individual error messages. The script is also connected to 
		the File Administration class, which properly administrates the entire 
        pipeline's inputs and outputs.
	'''
	def __init__(self):
		self.separator = 0
		self.file_count = 0

# ******************************************************************************
# ********************* Convolutional Neural Network ***************************
# ******************************************************************************

	def makeRandomGradient(self,size):
		''' Creates a random gradient

		Args:
			size: the size of the gradient

		Returns:
			the random gradient.
		'''
		x, y = np.meshgrid(np.linspace(0, 1, size[1]), 
						   np.linspace(0, 1, size[0]))
		grad = x * np.random.uniform(-1, 1) + y * np.random.uniform(-1, 1)
		grad = (grad - grad.mean()) / grad.std()
		return grad

	def alter(self,x):
		''' Applies a random gradient to the image

		Args:
			x: an image represented by a 3d numpy array

		Returns:
			the image with the added random gradient.
		'''
		grad = self.makeRandomGradient(x.shape)

		for i in range(3):
			x[:, :, i] = x[:, :, i] * np.random.uniform(0.9, 1.1)
		x = (x - x.mean()) / x.std()

		amount = np.random.uniform(0.05, 0.15)

		for i in range(3):
			x[:, :, i] = x[:, :, i] * (1 - amount) + grad * amount
		x = (x - x.mean()) / x.std()

		return x

	def additive_noise(self,x):
		''' Adds gaussian noise centered on 0 to an image.

		Args:
			x: an image represented by a 3d numpy array

		Returns:
			the image with the added noise.
		'''
		gauss = np.random.normal(0, 2 * 1e-2, x.shape)
		x = x + gauss
		return x

	def grayscale(self,x):
		''' Converts an image to grayscale.

		Args:
			x: an image represented by a 3d numpy array

		Returns:
			the grayscale image.
		'''
		return np.dstack([0.21 * x[:,:,2] + 0.72 * x[:,:,1] + 0.07 * x[:,:,0]] \
						 * 3)

	def flip(self,x, y):
		''' Flips an image and the corresponding labels.

		Args:
			x: an image represented by a 3d numpy array
			y: a list of labels associated with the image

		Returns:
			the flipped image and labels.
		'''
		if np.random.choice([True, False]):
			x = np.fliplr(x)

			for i in range(len(y) // 5):
				y[i * 5:(i + 1) * 5] = np.flipud(y[i * 5:(i + 1) * 5])

		return (x, y)

	def random_augment(self,im):

		choice = np.random.randint(0, 3)

		if choice == 0:
			im = self.additive_noise(im)
		elif choice == 1:
			im = self.grayscale(im)

		im = (im - im.mean()) / im.std()

		im = self.alter(im)

		return im

	def generator(self, chosen_dataset, group, batch_size, is_testset=False, 
				  augment=True, do_flip=True):
		''' Loads the dataset, preprocess it and generates batches of data.

		Args:
			chosen_dataset:	Path ot the dataset chosen
			group: 			List of ids from which to generate the data.
			batch_size: 	The size of the batch.
			is_testset: 	A boolean flag representing if the genrated data 
							will be used as test-set.
			augment: 		A boolean flag representing wether to augment the 
							data or not.
			do_flip: 		A boolean flag representing wether to also flip 
							horizontally the images and the labels.
						
		Returns:
			Preprocessed batches.
		'''
		h5f = h5py.File(chosen_dataset, 'r')
		Xs = {i: h5f['bag' + str(i) +'_x'] for i in group}
		Ys = {i: h5f['bag' + str(i) +'_y'] for i in group}
		lengths = {i: Xs[i].shape[0] for i in group}
		counts = {i: 0 for i in group}

		if is_testset and len(group) == 1:
			x = Xs[group[0]][:]
			y = Ys[group[0]][:]

			if do_flip:
				for i in range(x.shape[0]):
					x[i], y[i] = self.flip(x[i], y[i])

			y[y > 0] = 1.0
			mask = y != -1.0

			yield (x, y, mask)

		else:
			while True:
				index = np.random.choice(group)
				x = Xs[index][counts[index]:counts[index] + batch_size]
				y = Ys[index][counts[index]:counts[index] + batch_size]
				
				counts[index] += batch_size
				
				if counts[index] + batch_size > lengths[index]:
					counts[index] = 0

				if augment:
					for i in range(x.shape[0]):
						x[i] = self.random_augment(x[i])

				if do_flip:
					for i in range(x.shape[0]):
						x[i], y[i] = self.flip(x[i], y[i])

				y[y > 0] = 1.0 # binary classes 0, 1 and -1 for missing value

				yield (x, y)

	def model(self, target_count, lr=0.001, show_summary=False, 
			  old_version=True):
		'''Creates the keras neural network model.

		Args:
			lr: 			the learning rate used for the training.
			show_summary: 	a boolean flag that represents if the model has to 
							be printed to console.
			old_version: 	a boolean flag that represents if the model should 
							be the old one or the new one (more neurons).

		Returns:
			The defined keras model.
		'''
		model = Sequential()

		model.add(Conv2D(10, (3, 3), padding='same', input_shape=(64, 80, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(10, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(8, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())

		if old_version == True:
			model.add(Dense(32))
		else:
			model.add(Dropout(0.2))
			model.add(Dense(256))

		model.add(Activation('relu'))
		if old_version:
			model.add(Dense(20))
		else:
			model.add(Dense(target_count * 5))
		model.add(Activation('sigmoid', name='output'))

		def masked_mse(target, pred):
			mask = K.cast(K.not_equal(target, -1), K.floatx())
			mse = K.mean(K.square((pred - target) * mask))
			return mse

		model.compile(loss=masked_mse, optimizer=Adam(lr=lr))

		if show_summary:
			model.summary()

		return model

# ******************************************************************************
# ************************* Parameter Setup ************************************
# ******************************************************************************
	
	def calculate_separator(self, file_count, ratio):
		return int(file_count * ratio)
	
	def set_validation_parameters(self, file_count, separator):
		self.file_count = file_count
		self.separator = separator

	def get_validation_parameters(self):
		return self.separator, self.file_count

	def safe_param(self,name,n_epochs,steps,
				   batch_size,learning_rate,ratio,FA,idx):
		''' Function sends the most important parameters to the File
            Administration class where a markdown is automatically created.
            A further description of each parameter can be seen in the
            dictionary sent.
        '''
		header = "## Training Model \r\n"
		model_name = name + str(idx) + '.h5'
		loss_graph = '![Val_vs_Training](../../3_models/models_' + str(idx) \
																 + '/graph_' \
																 + str(idx) \
																 +'.png)'
		parameter_dict = {
			'- Model name: %s\r\n': model_name,
			'- No. of epochs: %s\r\n' : n_epochs,
			'- Steps: %s\r\n' : steps,
			'- Batch Size: %s\r\n' : batch_size,
			'- Learning Rate: %s\r\n' : learning_rate,
			'- Ratio val vs training: %s\r\n' : ratio,
			'### Validation vs Training loss \r\n %s \r\n' : loss_graph
		}
		FA.create_subsequent_markdown(parameter_dict,idx,header)


		header_latex = '\\textbf{Training Model}\r\n'
		model_name_latex = 'model\_' + str(idx) + '.h5'

		parameter_dict_latex = {
			'\\item  Model name: %s\r\n': model_name_latex,
			'\\item  No. of epochs: %s\r\n' : n_epochs,
			'\\item  Steps: %s\r\n' : steps,
			'\\item  Batch Size: %s\r\n' : batch_size,
			'\\item  Learning Rate: %s\r\n' : learning_rate,
			'\\item  Ratio val vs training: %s\r\n' : ratio,
		}
		FA.create_subsequent_latex(parameter_dict_latex,idx,header_latex)

# ******************************************************************************
# ********************************* Main ***************************************
# ******************************************************************************

	def train(self):
		''' Train the neural network model, save the weights and shows the
			learning error over time.
		'''
		parser = argparse.ArgumentParser()

		parser.add_argument('-n', '--name', type=str, 
							help='name of the Model weights', 
							default='model_')
		parser.add_argument('-e', '--epochs', type=int, 
							help='number of epochs of the training phase', 
							default=2)
		parser.add_argument('-s', '--steps', type=int, 
							help='number of training steps per epoch', 
							default=10)
		parser.add_argument('-bs', '--batch-size', type=int, 
							help='size of the batches of the training data', 
							default=64)
		parser.add_argument('-lr', '--learning-rate', type=float, 
							help='learning rate used for the training phase', 
							default=0.0004)
		parser.add_argument('-tv', '--ratio',type=float,
							help='ratio of training and validation set',
							default=0.85)
		parser.add_argument('-m', '--mode',type=int,
							help='autonomous [0] or normal mode [1]',
							default=1)
		parser.add_argument('-tc','--target-count',type=int,
                            help='amount of positions in dm of which to relate',
                            default=35)
		args = parser.parse_args()

		name = args.name
		n_epochs = args.epochs
		steps = args.steps
		batch_size = args.batch_size
		learning_rate = args.learning_rate
		ratio = args.ratio
		mode = args.mode
		target_count = args.target_count

		np.set_printoptions(suppress=True)
		np.set_printoptions(precision=2)

		print()
		print('Parameters:')
		for k, v in vars(args).items():
			print(k, '=', v)
		print()

		FA = FileAdministration()
		idx = FA.get_pipeline_index(current=True)
		bagfile_dir = FA.get_pipeline_directory(1,idx)
		dataset_dir = FA.get_pipeline_directory(2,idx)
		model_dir = FA.get_pipeline_directory(3,idx)
		file_count = FA.get_file_count(bagfile_dir)
		
		
		self.safe_param(name,n_epochs,steps,
		 				batch_size,learning_rate,ratio,FA,idx)

		separator = self.calculate_separator(file_count,ratio)
		
		cnn = self.model(target_count, learning_rate, show_summary=True, 
						 old_version=False)

		gen = self.generator(dataset_dir +'dataset_'+str(idx)+'.h5',
							 np.arange(0, separator), 
							 batch_size, 
							 is_testset=False, 
						 	 augment=True, 
							 do_flip=True)

		validation = next(self.generator(dataset_dir +'dataset_'+str(idx)+'.h5',
							np.arange(separator,file_count), 
							1000, 
							is_testset=False, 
							augment=True, 
							do_flip=True))

		history = cnn.fit_generator(generator=gen, 
									steps_per_epoch=steps, 
									epochs=n_epochs,
									validation_data = validation)

		filename = model_dir + name + str(idx) + '.h5'

		cnn.save_weights(filename)

		l = len(history.history)
		print("Length history")
		print(l)
		val = np.empty
		for i, t in enumerate(history.history.items()):
			plt.subplot(1,l, i + 1)
			plt.plot(t[1])
			plt.title(t[0])
			print(t[0])
			print(t[1])
			if(i == 0):
				val = t[1]
			else:
				train = t[1]
			print('-------------------')
		
		#Provide space between subplots
		plt.subplots_adjust(wspace=0.3) 
		plt.savefig(model_dir + 'graph_' + str(idx) + '.png',dpi = 100,bbox_inches='tight')
		
		#Generalization plot
		gen_array = np.subtract(val,train)
		plt.subplot(1,1,1)
		plt.plot(gen_array)
		plt.title('Generalization Error')
		plt.savefig(model_dir + 'gen_loss_' + str(idx) + '.png',dpi = 100,bbox_inches='tight')

if __name__ == '__main__':
	t = Train()
	t.train()