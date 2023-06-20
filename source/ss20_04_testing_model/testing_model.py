import os
import sys
import tqdm
import pickle
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import roc_auc_score, roc_curve
from source.ss20_03_training_model.training_model import Train
from source.files.file_administration_v_02 import FileAdministration

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
			return 'CustomError, {0}{1} !'.format(self.message,self.index)
		else:
			return 'CustomError has been raised'



class Test():
	''' At this step, the model is being tested against the test data of a 
		dataset. The user will be able to select a trained model, if several 
		models are available. Once a model is chosen, the related dataset and 
		its testset are loaded for testing. 
	'''
	def safe_param(self,rounds,FA,idx):
		''' Function sends the most important parameters to the File
            Administration class where a markdown is automatically created.
            A further description of each parameter can be seen in the
            dictionary sent.
        '''
		header = "## Testing Model \r\n"
		auc_graph_1 = '![AUC](../../4_plots/plots_' + str(idx) \
													+ '/AUCvsDist_' \
													+ str(idx) \
													+ '.png)'
		auc_graph_2 = '![AUC](../../4_plots/plots_' + str(idx) \
													+ '/AUC_' \
													+ str(idx) \
													+ '.png)'
		parameter_dict = {
			'- Rounds: %s\r\n': rounds,
			'### Area under the Curve \r\n %s' : auc_graph_2
		}
		FA.create_subsequent_markdown(parameter_dict,idx,header)

		header_latex = '\\textbf{Testing Model}\r\n'
		parameter_dict_latex = {
			'\\item Rounds: %s\r\n': rounds
		}
		FA.create_subsequent_latex(parameter_dict_latex,idx,header_latex,type=1)
		FA.create_last_latex(idx)

	def test(self):
		''' Following code represents four loops where the outer loop 
			iterates over the valdiation set. The size of the validation set,
			and therefore the size of the outer loop depends on how big the
			entire dataset is. The default ratio is 0.8. 
			
			The second loop iterates 100 times over the both inner subloops.

			The two inner subloops actually do the calculation. The goal here is
			to create a (32,5) numpy array with AUC values from test_y (shape
			60,155) and its corresponding predictions made from the images.
			
			Ground Truth:
			The test_y represents the ground truth in a 2 dimensional array
			with 60 rows and 155 columns. Every row represents one input. Every
			column is representing one of the five laser ranges 31 times where
			each set of 5 lasers represent a distance up to 31.
			Example:
			Column 1-5 represent 5 different laser ranges and belong to distance
			of 1 meter.
			Column 6 to 10 also represent 5 different laser ranges but belong
			to distance of meter 2.

			Calculation:
			As the first step, just observed values are extracted from every
			column of test_y. This is done with help of a mask returned by
			the generator. The values are then randomly chosen in a next step.
			To fill the (32,5) numpy array with AUC, the ground truth test_y is
			evaluated against the prediction for all input values as once as
			with iteration through every column at the most inner loop. For 
			each column, which represents here 60 input values, one AUC value
			is calculated by observing the ground truth and the predictions.
			This value is then copied to the (32,5) numpy array starting
			with the distance of 1 meter and the first laser range respectively.
			The inner loop therefore fills 1 row each five iterations all
			corresponding to the first meter of distance. The second inner loop
			then calculates the second meter respectively.
		'''

		parser = argparse.ArgumentParser()
		parser.add_argument('-m', '--mode',type=int,
							help='autonomous [0] or normal mode [1]',
							default=1)
		parser.add_argument('-r', '--rounds',type=int,
							help='number of rounds',
							default=100)
		parser.add_argument('-d', '--n-dist',type=int,
							help='number of distances',
							default=35)
		parser.add_argument('-tv', '--ratio',type=float,
							help='ratio of training and validation set',
							default=0.85)
		args = parser.parse_args()
		mode = args.mode
		rounds = args.rounds
		n_dist = args.n_dist
		ratio = args.ratio

		np.set_printoptions(suppress=True)

		distances = list(range(0, n_dist, 1))

		print('number of rounds = %d' % rounds)

		auc_array = []

		t = Train()

		cnn = t.model(n_dist,show_summary=True, old_version=False)

		FA = FileAdministration()
		idx = FA.get_pipeline_index(current=True)
		bagfile_dir = FA.get_pipeline_directory(1,idx)
		dataset_dir = FA.get_pipeline_directory(2,idx)
		model_dir = FA.get_pipeline_directory(3,idx)
		plot_dir = FA.get_pipeline_directory(4,idx)
		file_count = FA.get_file_count(bagfile_dir)

		self.safe_param(rounds,FA,idx)

		separator = int(file_count * ratio)

		cnn.load_weights(model_dir + 'model_' + str(idx) + '.h5')

		rng = np.random.RandomState(13)

		for i in np.arange(separator,file_count):
			print('Testing on Validation: ' + str(i) + ' from: ' + str(file_count) )
			group = [i]
			''' Each value in the mask is equal to 1 (True) if the corresponding
				label is known (0 or 1) and equal to 0 (False) if the label is
				unknown.
			'''
			test_x, test_y, mask = next(t.generator(dataset_dir \
													+ 'dataset_' \
													+ str(idx) \
													+'.h5',
													group, 
													32, 
													is_testset=True, 
													augment=False, 
													do_flip=True))

			loss = cnn.evaluate(test_x, test_y, verbose=1)
			prediction = cnn.predict(test_x)
			#unique, counts = np.unique(prediction[50], return_counts=True)
			for p in tqdm.tqdm(range(rounds)):#rounds == 100
				aucs = np.zeros([n_dist, 5])
				for i, d in enumerate(distances):
					d = '%.1f' % d
					for j in range(5):
						indices = np.where(mask[:, i * 5 + j])
	
						if len(indices[0]) > 0:
							indices = (rng.choice(indices[0], len(indices[0])),)
						try:
							''' roc_auc_score gets as parameter values from
								test_y. The inner loop will loop through each
								column of test_y five times. test_y represents 
								the ground truth and has in total 150 columns
								and 60 rows. indices is a numpy array
								containing indices pointing towards positive
								values of the mask array returned from the
								generator. Each value in indices is applied
								on each row selecting the row which the 
								value of indices is pointing to. At the end
								the nd array is transferred to a regular list.
								The same procedure is applied to the prediction.

								Example for small example array:
								b = np.random.rand(3,2)
								b  ([[0.07062192, 0.64180385],
       							     [0.17434824, 0.04706346],
       							     [0.72906399, 0.94646209]])
								c = np.random.randint(3,size=(1,3))
								c ([[0, 1, 1]])
								Result of b[c,1]
								array([[0.64180385, 0.04706346, 0.04706346]])

								The function in general calculates the AUC 
								(Area under the Curve) for each binary ground
								truth value column related to its corresponding
								predictions column.
								
								Example:
								import numpy as np
								from sklearn.metrics import roc_auc_score
								y_true = np.array([0, 0, 1, 1])
								y_scores = np.array([0.1, 0.4, 0.35, 0.8])
								roc_auc_score(y_true, y_scores)
								0.75

							'''
							auc = roc_auc_score(test_y[indices, i * 5 + j] \
											    .tolist()[0], 
												prediction[indices, i * 5 + j] \
												.tolist()[0])

						except ValueError as e:
							auc = 0.5
						aucs[n_dist - 1 - i, j] = auc

				auc_array.append(aucs)
		mean_auc = np.mean(auc_array, axis=0)
		std_auc = np.std(auc_array, axis=0)

		''' Area under the Curve
		'''
		dist_labels = ['%.0f' % d for d in np.flipud(distances)]
		dist_labels = [d for i, d in enumerate(dist_labels) if i % 2 == 0]
		colors = ['aqua', 'darkorange', 'deeppink', 'cornflowerblue', 'green']

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.75, 3.8))
		tmp = mean_auc *100
		
		'''
		Makes the array n times smaller by calculating the mean of every fourth 
		row to properly display the heatmap. Makes the amount of rows uneven. 
		This calculation just needs to be done, if the number of rows is too big 
		for the heatmap to be properly displayed.
		'''

		if(n_dist == 81):				  
			tmp = np.delete(tmp,0,0)
			tmp = 0.5*(tmp[0::4] + tmp[1::4])
			dist_labels = np.arange(len(tmp)*2,0,-2)
		if(n_dist == 41 or n_dist == 35):
			tmp = np.delete(tmp,0,0)
			tmp = 0.5*(tmp[0::2] + tmp[1::2])
			dist_labels = np.arange(len(tmp)*2,0,-2)
		newArray = tmp.astype(int)
		sns.heatmap(newArray, 
					cmap='Blues', 
					annot=True, 
					vmin=50, 
					vmax=100, 
					annot_kws={'color':'white'},
					fmt='d')
		ranges = []
		SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
		for i in range(5,0,-1):
			ranges.append('r'+ str(i).translate(SUB))
		ax.set_xticklabels(ranges)
		ax.set_yticklabels(dist_labels, rotation=0)
		plt.xlabel('Ranges')
		plt.ylabel('Distance [dm]')
		plt.title('Area under the curve')
		plt.savefig(plot_dir + 'AUC_'+ str(idx) +'.png', 
					dpi = 100,
					bbox_inches='tight')
		#plt.show()

if __name__ == '__main__':
	t = Test()
	t.test()