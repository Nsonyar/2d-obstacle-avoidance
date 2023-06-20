import os
import cv2
import h5py
import rosbag
import argparse
import sys, getopt
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as Image_1
from file_administration_v_02 import FileAdministration
from source.ss20_02_feature_extraction.feature_extraction import FeatureExtraction

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

class Develop:
    ''' Class containing testing, debugging and developer functions related to 
        differnt parts of the pipeline. Functions are not actively used in the
        pipeline.
    '''
    def __init__(self,name_of_file = None, args = None):
        ''' Init function
        Args:
            optional argument of location and name of where the file is read 
            from
        '''
        self.name = name_of_file
        self.args = args

    def get_base_directory(self):
        ''' Dataset specific function extracting base directory and more
            items inside the dataset. Needs to be tested further.
        '''
        with h5py.File('./2_datasets/datasets_8/dataset_8.h5','r') as hdf:
            base_items = list(hdf.items())
            n1 = hdf.get('bag20_x')
            n1 = np.array(n1)
            print(n1.shape)
            print(n1[0])
            print(type(n1[0]))
            print(n1[0].shape)
            img = Image_1.fromarray(n1[59], 'RGB')
            img.save('my.png')

    def get_group_key(self):
        ''' Function printing out group keys and further data given the keys
        '''
        with h5py.File(self.name, 'r') as f:
        
            # List all groups
            print('Keys: %s' % f.keys())
            print(f.items())
            a_group_key = list(f.keys())[2]
            print(a_group_key)

            # Get the data
            data = list(f[a_group_key])
            print(len(data))
    
    def get_function_array(self):
        ''' Function importing strings of function names in order to test them
            for exceptions in a loop.
        '''
        FE = FeatureExtraction()
        array = dir(FeatureExtraction)
        array = ['FE.'+ i + '(1)' for i in array]
        for i in range(17,18):
            try:
                eval(array[i])
            except Exception:
                print(Exception)
    
    def test_string_to_np(self):
        ''' Function used to invididually test another function for exceptions.
        '''
        FE = FeatureExtraction()
        try:
            FE.get_img_as_np(1)
        except Exception:
            
            print(Exception)
        
    def get_img_as_np(self, image, size=None, normalize=False):
        ''' Converts a jpeg image in a 2d numpy array of BGR pixels
            and resizes it to the size given.

        Args:
            image:  a compressed BGR jpeg image.
            size:   a tuple containing width and height, or None for 
                    no resizing.

        Returns:
            The raw, resized image as a 2d numpy array of BGR pixels.
        '''
        if(len(image)<100):
            raise CustomError('Image has wrong length. Check manually !')

        compressed = np.fromstring(image, np.uint8)
        raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        if size:
            raw = cv2.resize(raw, size)
        if normalize:
            raw = (raw - raw.mean()) / raw.std()
        
        img = Image_1.fromarray(raw, 'RGB')
        img.save('my.png')
        return raw

    def read_bag_file(self):
        ''' Function used to read out bagfiles information
        '''
        bag = rosbag.Bag('./1_bagfiles/bagfiles_53/[10]_53.bag')
        counter = 0
        for topic, msg, t in bag.read_messages(topics=['image']):
            print(type(msg))
            print(type(t))
            print(topic)
            print(msg.data)
            print(t)
        #     counter = counter + 1
        #     if(counter == 8):
        #         self.get_img_as_np(msg.data, (80, 64))
        #         break      
        #for topic, msg, t in bag.read_messages(topics=['odom']):
            #print(type(topic))
            #print(type(msg))
            #print(type(t))
            #print(topic)
            #print(msg.pose.pose.position.x)
            #print(t)
        # for topic, msg, t in bag.read_messages(topics=['laser']):
        #     print(type(topic))
        #     print(type(msg))
        #     print(type(t))
        #     print(topic)
        #     print(msg)
        #     print(t)
        bag.close()

    def dataset_test(self):
        ''' Testing function to read out a h5f dataset
        '''
        group = [1]
        h5f = h5py.File('./2_datasets/datasets_8/dataset_8.h5', 'r')

        Xs = {6: h5f['bag' + str(6) +'_x'] }
        Ys = {6: h5f['bag' + str(6) +'_y'] }

        print('X Values')
        print(len(Xs))
        print(Xs.shape)
        print(type(Xs))
        print(Xs)
        print(Xs.items())
        
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

    def image(self):
        ''' Displaying png image with cv2
        '''
        img = cv2.imread('AUC_1.png',0)
        cv2.imshow('Test',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def matplot(self):
        ''' Tick tester function
        '''
        img = cv2.imread('AUC_1.png')
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])
        plt.show()

    def getopt(self):
        ''' C style parameter setup. Argparsed mostly used currently
        '''
        inputfile = ''
        outputfile = ''
        try:
            opts, args = getopt.getopt(self.args,'hi:o:',['ifile=','ofile='])
        except getopt.GetoptError:
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('test.py -i <inputfile> -o <outputfile>')
                sys.exit()
            elif opt in ('-i', '--ifile'):
                inputfile = arg
            elif opt in ('-o', '--ofile'):
                outputfile = arg
        print('Input file is ', inputfile)
        print('Output file is ', outputfile)

    def get_ranges(self,total_range,sections,laser_count):
        ''' Test function to evenly divide laser
        Args:
            total_range:    number of lasers to take in consideration
            sections:       number of total range to be divided by
            laser_count:    robot specific amount of lasers
        '''
        invdividual_range = int(round(total_range / 5))
        temp_index = int(round((laser_count - total_range) / 2))
        arr = np.empty(0,int)
        for i in range(sections):
            for j in range(2):
                arr = np.append(arr,temp_index)
                temp_index = temp_index + invdividual_range
            temp_index = temp_index - invdividual_range + 1
        print(type(arr[0]))
        print(arr)

    def create_first_markdown(self,arr):
        ''' Test function for markdown creation
        Args:
            arr: List of string to fill markdown file with
        '''
        if(os.path.exists('./parameters/')):
            print('Folder exist')
        else:
            os.makedirs('./parameters/')
        
        index = 1
        if(os.path.exists('./parameters/results['+str(index)+'].md')):
            print('Custom Error at 1 function!')
            sys.exit(1)
        else:
            f= open('./parameters/results['+str(index)+'].md','w+')
            f.write('Data Acquisition\r\n')
            for i in range(len(arr)):
                f.write('- %s\r\n' % (arr[i]))
            f.close() 

    def create_subsequent_markdown(self):
        ''' Test function for adding text to existing markdown
        '''
        index = 1
        if(os.path.exists('./parameters/results['+str(index)+'].md')):
            f= open('./parameters/results['+str(index)+'].md','a+')
            for i in range(10):
                f.write('# This is line appended %d\r\n' % (i+100))
            
            f.write('subsquent text')
            f.close()
        else:
            print('Custom Error !')
            sys.exit(1)

    def array_for_markdown(self):
        ''' Test function to initialize markdown creation
        '''
        arr = ['name1','name2','name3']
        self.create_first_markdown(arr)
        self.create_subsequent_markdown()

    def customError(self):
        ''' Test fnction for CustomError function
        '''
        try:
            print(askkas)
        except:
            idx = '5'
            raise CustomError("test",idx)
    
    def subplot(self):
        ''' Creating of subplots
        '''
        # Some example data to display
        x = np.linspace(0, 2 * np.pi, 400)
        y = np.sin(x ** 1)
        plt.subplots_adjust(wspace=2)        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Horizontally stacked subplots')
        ax1.plot(x, y)
        ax2.plot(x, -y)
        fig = plt.gcf() 
        plt.rc('font', size=12)
        fig.set_size_inches(8, 6)
        plt.subplots_adjust(wspace=0.3)    
        plt.savefig("myplot.png", dpi = 100)
        #plt.show()

    def image_interactive(self):
        ''' Creation of interactive heatmap with matplotlib out of np array
        '''
        IMAGE_SIZE = 20
        plt.ion()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        # this example doesn't work because array only contains zeroes
        array = np.zeros(shape=(10, IMAGE_SIZE), dtype=np.uint8)
        axim1 = ax1.imshow(array)
        # this value allow imshow to initialise it's color scale
        array[0, 0] = 99 
        axim2 = ax2.imshow(array)
        del array
        for _ in range(50):
            print('.', end='')
            matrix = np.random.randint(0, 
                                       2, 
                                       size=(IMAGE_SIZE, IMAGE_SIZE), 
                                       dtype=np.uint8)
            print(matrix.shape)
            axim1.set_data(matrix)
            fig1.canvas.flush_events()
            
            axim2.set_data(matrix)
            fig2.canvas.flush_events()
        print()

    def aspect_ratio(self):
        ''' Stretching matplot image keeping the aspect ratio
        '''
        plt.style.use('ggplot')
        x = np.linspace(-5, 5, 100)
        y1 = np.exp(0.8*x)
        y2 = np.sin(x)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y1)
        ax.plot(x, y2)

        ratio = 0.3
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        # the abs method is used to make sure that all numbers are positive
        # because x and y axis of an axes maybe inversed.
        #ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

        # or we can utilise the get_data_ratio method which is more concise
        ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
        plt.show()
    
    def create_empty_pipeline(self):
        ''' Function can create an empty pipeline index folder structure
        '''
        FA = FileAdministration()
        FA.create_pipeline_directories(62)

if __name__ == '__main__':
    t = Develop()
    t.create_empty_pipeline()

    #t.read_bag_file()
    #t.get_ranges(200,5,666)
    



