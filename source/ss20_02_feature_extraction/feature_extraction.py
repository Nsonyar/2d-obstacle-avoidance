import os
import re
import sys
import cv2
import tqdm
import h5py
import tqdm
import types
import rosbag
import shutil
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from datetime import timedelta
from scipy.spatial.transform import Rotation as R
from source.files.file_administration_v_02 import FileAdministration

# ******************************************************************************
# *************************** Custom Error class *******************************
# ******************************************************************************

class CustomError(Exception):
	''' Custom Class to provide the user with a proper error message in case
		some exception is raised.
	
	Args:
		msg: 	optional parameter to provide a message to the user
		index: 	optional parameter to indicate further index information about
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

class FeatureExtraction:
    ''' As the second stage of this pipeline, this script take bagfiles as input
        and creates a dataset for training. Some concepts and snippets
        mostly from the self-supervised learning part of the script, have been
        provided by [NGCG+19] as mentioned in the paper. It is recommended to
        start the script from the pipeline launcher, which incorporates it
        within the project. Parameters can be set manually or through the
        pipeline launcher. The script contains preprocessing mechanisms,
        self-supervised learning techniques, a custom error class to print
        individual messages, error recognition mechanism and a parameter setup
        for fully autonomous documentation. The script is also connected to the
        File Administration class, which properly administrates the entire 
        pipeline's inputs and outputs.
    '''
# ******************************************************************************
# ************************* Set global variables *******************************
# ******************************************************************************
    def set_target_positions_arr(self, target_count):
        ''' Set of global variable used in several instances
        Args:
            target_count: amount of positions in dm of which to relate from
        '''
        self.target_count = target_count
        self.target_positions_arr = np.arange(target_count,dtype=np.float)

# ******************************************************************************
# ************************* Preprocessing **************************************
# ******************************************************************************

    def get_ranges(self,total_range,sections,laser_count):
        ''' Function calculates an array which is used to extract a specific
            amount of data from laser recording.
        Args:
            total_range:    total amount of lasers to be taken in consideration
            sections:       amount of sections total_range is divided by
            laser_count:    total amoutn of laserscanners. Robot specific
        Return:
            numpy array of indexes to divide the laser with
        '''
        invdividual_range = int(round(total_range / sections))
        temp_index = int(round((laser_count - total_range) / 2))
        arr = np.empty(0,int)
        for i in range(sections):
            for j in range(2):
                arr = np.append(arr,temp_index)
                temp_index = temp_index + invdividual_range
            temp_index = temp_index - invdividual_range + 1
        return arr

    def get_dictionary(self, range_arr):
        ''' Getter function for dictionary which can be applied to bag
            file in order to extract relevant camera, laser and
            positional data
        '''
        if(isinstance(range_arr,np.ndarray) != True):
            raise CustomError('Variable range_arr has wrong type !')

        dictionary = {
            'image': lambda m: 
            {
                'camera': self.get_img_as_np(m.data, (80, 64))
            },
            'laser': lambda m: 
            {
                'r1': self.calc_bin_label(m.ranges,range_arr[0],range_arr[1]),
                'r2': self.calc_bin_label(m.ranges,range_arr[2],range_arr[3]),
                'r3': self.calc_bin_label(m.ranges,range_arr[4],range_arr[5]),
                'r4': self.calc_bin_label(m.ranges,range_arr[6],range_arr[7]),
                'r5': self.calc_bin_label(m.ranges,range_arr[8],range_arr[9]),
            },
            'odom': lambda m: 
            {
                'x': m.pose.pose.position.x,
                'y': m.pose.pose.position.y,
                'theta': self.get_yaw_from_quaternion(m.pose.pose.orientation)
            }
        }
        return dictionary
    
    def calc_bin_label(self, msg, range_start, range_end):
        ''' Extracting laser values from specific ranges returning a
            binary value whether laser was below a certain threshold
            or not.

        Args:
            msg:            ros message contained in bag file
            range_start:    start of laser index taken in consideration
            range_end:      end of laser index taken in consideration
        '''
        ranges = msg[range_start:range_end]
        for i in range(len(ranges)):
            if ranges[i] < self.laser_threshold:
                return 1
        return 0

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
        return raw

    def get_yaw_from_quaternion(self,q):
        ''' Calculation of euler values given quaternions

        Args:
            q: quaternion values

        Return:
            yaw from euler values
        '''
        r = R.from_quat([q.x,q.y,q.z,q.w])
        return r.as_euler('zyx', degrees=False)[0]

    def get_df_prototype(self, bag, dictionary):
        ''' Creation of dataframe prototype for each rostopic

        Args:
            bags:       current rosbag file
            dictionary: rostopics with functions to extract relevant 
                        values

        Return:
            dictionray of dataframes by ros topic
        '''
        result = {}

        for topic in dictionary.keys():
            timestamps = []
            values = []

            for subtopic, msg, t in bag.read_messages(topic):
                if subtopic == topic:
                    timestamps.append(msg.header.stamp.to_nsec())
                    values.append(dictionary[topic](msg))

            df = pd.DataFrame(data=values, 
                              index=timestamps, 
                              columns=values[0].keys())
            result[topic] = df
        return result

    def get_df_syncronized(self,dfs):
        ''' Syncronization of different dataframes into one

        Args:
            dfs: dataframes as dictionary divided by ros topic

        Return:
            Syncronized dataframe
        '''
        min_topic = None

        for topic, df in dfs.items():
            if not min_topic or len(dfs[min_topic]) > len(df):
                min_topic = topic

        values = []
        ref_df = dfs[min_topic]
        # Remove bagfiles with the same time
        len_with_duplicates = len(ref_df)
        ref_df = ref_df[~ref_df.index.duplicated(keep='first')]
        len_without_duplicates = len(ref_df)
        dif = len_with_duplicates - len_without_duplicates
        if(dif > 5):
            raise CustomError('More than 5  df.index duplicates. Please '
                              'manually check bagfiles !')
        elif(dif < 5):
            print('Duplicates: ' + str(dif))

        topics = list(dfs)  
        topics.remove(min_topic)

        topics = [(0, topics[0]), (0, topics[1])]

        for i in tqdm.tqdm(range(0, len(ref_df)), 
                           desc='generating datapoints'):

            t = ref_df.index[i]
            row =[{'timestamp': t}, ref_df.loc[t].to_dict()]
            for i, topic in topics:
                df = dfs[topic]
                
                while i < len(df) and df.index[i] < t:
                    i += 1
                
                if i >= len(df):
                    row = None
                    break

                row.append(df.iloc[i].to_dict())

            if row:
                values.append({k: v for d in row for k, v in d.items()})

        result = pd.DataFrame.from_dict(values).set_index('timestamp')
        result.index = pd.to_datetime(result.index)
        return result

# ******************************************************************************
# ************************* Self-Supervised Learning ***************************
# ******************************************************************************

    def value_inside_range(self, x, range):
        return np.logical_and(range[0] < x,  x < range[1])

    def check_target_validity(self,target_position, dx, dy, dtheta):
        ''' Checks if positional values of relative distances are within
            a specific range. If they are, return values for function
            value_inside_range will turn to true and corresponding laser
            values of current dataframe selection row will be connected
            to current of entire dataframe

        Args:
            target_position:    current target position
            dx:                 x values of relative distances calculated 
                                from current df selection
            dy:                 y values of relative distances calculated 
                                from current df selection
            dtheta:             difference of theta values from dataframe 
                                selection and current row

        Return:
            Boolean array with size of current dataframe selection 
            indicating with its values which rows of dataframe selection 
            are being filtered
        '''
        #target_position /= self.target_count possible erroneous
        ''' target_position needs to be expresed in meters as it is given in
            decimeters. If divided by 10 might not be sufficient. A higher value
            like 11.5 or 12 can also be tested with.
        '''
        target_position = target_position / 12
        
        return  self.value_inside_range(dx, (target_position - 0.07, 
                                             target_position + 0.07)) & \
                self.value_inside_range(dy, (-0.07, 0.07)) & \
                self.value_inside_range(dtheta, (-0.3, 0.3))

    def get_ground_truth_to_target(self,
                                   relative_distances,
                                   dataframe_selection,row,
                                   target_position):
        ''' Getting dataframe with ground truth laser values which are
            close to current target position

        Args:
            relative_distances:     np array with positional values 
                                    transformed to current's row 
                                    coordinate system
            dataframe_selection:    current dataframe selection
            target_position:        current target position

        Return: 
            Filtered dataframe_selection with values close to 
            target_position
        '''
        dx = relative_distances[:, 0] 
        dy = relative_distances[:, 1] 
        dtheta = (dataframe_selection['theta'] - row['theta']).values 
        
        return dataframe_selection[self.check_target_validity(target_position, 
                                                              dx, 
                                                              dy, 
                                                              dtheta)]

    def transformation_matrix(self, positional_value, row):
        ''' Transformation of positional value to coordinate system of 
            row

        Args:
            positional_value:   one positional value from current 
                                dataframe selection
            row:                current row considered as point 
                                of origin

        Return:
            Numpy array with one positional value transformed to 
            current's row coordinate system
        '''
        cos = np.cos(row['theta'])
        sin = np.sin(row['theta'])
        x = row['x']
        y = row['y']
        inverse_frame = np.linalg.inv(np.array([[cos, -sin, x], 
                                               [sin, cos, y], 
                                               [0, 0, 1]]))
        return np.matmul(inverse_frame, positional_value)
    
    def get_relative_distances(self,positional_values,row,length):
        ''' Getting distances in relation to current row's positional 
            values

        Args:
            positional_values: array of positional values extracted from
            current dataframe selection
            row: current row considered as point of origin
            length: length of current dataframe selection

        Return:
            Numpy array with positional values transformed to current's
            row coordinate system
        '''
        return np.array([self.transformation_matrix(positional_values[i, :], 
                                                    row) 
                                                    for i in range(length)])

    def get_dataframe_selection_range(self, 
                                      df, 
                                      target_position, 
                                      i):
        ''' Function selecting a relevant range for calculation

        Args:
            df: relevant dataframe
            target_position: current position used for calculation
            i: current row of dataframe
            selection: type of selection used (1)==dynamic (2)==static 
        '''
        if(self.selection_range == 1):
            min_speed = 5
            max_speed = 40
            return df.loc[i + pd.Timedelta(str(round(target_position 
                                                       / max_speed, 1)) 
                                                       + 's'):
                            i + pd.Timedelta(str(round(target_position 
                                                         / min_speed, 1)) 
                                                         + 's')]
        else:
            td_wbegin = pd.Timedelta('-60 s')
            td_wend = pd.Timedelta('+60 s')
            return df.loc[i + td_wbegin:i + td_wend]

    def get_positional_values(self,dataframe_selection):
        ''' Extracting x and y value from the current dataframe
            selection

        Args:
            dataframe_selection: current dataframe selection

        Return:
            Numpy array with positional values from current 
            dataframe selection
        '''
        return np.concatenate([
            np.expand_dims(dataframe_selection['x'].values, axis=1),
            np.expand_dims(dataframe_selection['y'].values, axis=1),
            np.ones((len(dataframe_selection), 1))], axis=1)

    def relate_ground_truth(self, row, df, target_position):
        ''' This function relates given target positions to ground
            truth laser data. The function is called for each target
            position for every row in the dataframe filling in the
            approriate values. If target position for example is 5,
            the function is called for every row in the dataframe,
            trying to find values which are 5 steps away from the
            current rows positional values.

        Args:
            row: current row considered as point of origin
            df: entire dataframe
            target_position: current target position

        Return:
            Relative ground truth from lasers for current target
            position
        '''
        i = row.name
        prefix = 't_%.1f_' % target_position

        dataframe_selection = \
        self.get_dataframe_selection_range(df,
                                           target_position,
                                           i)

        if len(dataframe_selection) == 0:
            return pd.Series({
                prefix + 'r1': None,
                prefix + 'r2': None,
                prefix + 'r3': None,
                prefix + 'r4': None,
                prefix + 'r5': None
                })
        
        positional_values = \
        self.get_positional_values(dataframe_selection)

        relative_distances = \
        self.get_relative_distances(positional_values,
                                    row,len(dataframe_selection))

        ground_truth_to_target = \
        self.get_ground_truth_to_target(relative_distances,
                                        dataframe_selection,
                                        row,
                                        target_position)
        #print(len(ground_truth_to_target))
        if len(ground_truth_to_target) == 0:
            return pd.Series({
                prefix + 'r1': None,
                prefix + 'r2': None,
                prefix + 'r3': None,
                prefix + 'r4': None,
                prefix + 'r5': None
                })
        elif len(ground_truth_to_target) == 1:
            return pd.Series({
                prefix + 'r1': ground_truth_to_target['r1'][0],
                prefix + 'r2': ground_truth_to_target['r2'][0],
                prefix + 'r3': ground_truth_to_target['r3'][0],
                prefix + 'r4': ground_truth_to_target['r4'][0],
                prefix + 'r5': ground_truth_to_target['r5'][0]
                })
        else:
            return pd.Series({
                prefix + 'r1': ground_truth_to_target.iloc[0]['r1'],
                prefix + 'r2': ground_truth_to_target.iloc[0]['r2'],
                prefix + 'r3': ground_truth_to_target.iloc[0]['r3'],
                prefix + 'r4': ground_truth_to_target.iloc[0]['r4'],
                prefix + 'r5': ground_truth_to_target.iloc[0]['r5']
                })

    def get_df_with_groundTruth(self,df):

        for targ_pos in tqdm.tqdm(self.target_positions_arr, 
                                 desc='pull future readings'):

            next_laser = df.apply(self.relate_ground_truth, 
                                  axis=1, args=(df, targ_pos))
            df = pd.concat([df, next_laser], axis=1)
        return df

    def calc_camera(self,row):
        ''' Debug function replacing lambda mechanism. Currently not used.
            Function needs to be called in df.apply as first parameter.
        Args:
            row:    row pandas dataframe. 
        '''
        print(type(row))
        print()
        return (row - row.mean()) / row.std()

# ******************************************************************************
# ************************* Error recognition **********************************
# ******************************************************************************
    
    def set_selection_range(self, selection_range):
        ''' Set of global variable used in several instances
        Args:
            selection_range:    value to determine how positional values are
                                being used.
        '''
        if(isinstance(selection_range,int) != True):
            raise TypeError('Fix datatype used !')
        self.selection_range = selection_range
    
    def set_laser_threshold(self, laser_threshold):
        ''' Set of global variable used in several instances
        Args:
            laser_threshold: Range withing objects are relevant 
        '''
        if(isinstance(laser_threshold,float) != True):
            raise TypeError('Fix datatype used !')
        self.laser_threshold = laser_threshold

# ******************************************************************************
# ************************* Parameter Setup ************************************
# ******************************************************************************

    def safe_param(self,sections,laser_count,range_arr,
                   selection_range,target_count,laser_threshold,FA,idx):
        ''' Function sends the most important parameters to the File
            Administration class where a markdown and latex file are 
            automatically created. A further description of each parameter can 
            be seen in the dictionary sent.
        '''
        laser_range = str(range_arr[0]) + ':' + str(range_arr[len(range_arr)-1])
        dataset_name = 'dataset_' + str(idx) + '.h5'
        header = "## Feature Extraction \r\n"
        parameter_dict = {
            '- Dataset name: %s\r\n': dataset_name,
            '- laser range: %s\r\n': laser_range,
            '- laser sections: %s\r\n' : sections,
            '- laser count: %s\r\n' : laser_count,
            '- selection range: %s\r\n' : selection_range,
            '- target count: %s\r\n' : target_count,
            '- laser threshold: %s\r\n' : laser_threshold
        }
        FA.create_subsequent_markdown(parameter_dict,idx,header)

        header_latex = '\\textbf{Feature Extraction}\r\n'
        dataset_name_latex = 'dataset\_' + str(idx) + '.h5'
        laser_range_latex = str(range_arr[0]) \
                                + ':' \
                                + str(range_arr[len(range_arr)-1])
        parameter_dict_latex = {
            '\\item Dataset name: %s\r\n': dataset_name_latex,
            '\\item  laser range: %s\r\n': laser_range_latex,
            '\\item  laser sections: %s\r\n' : sections,
            '\\item  laser count: %s\r\n' : laser_count,
            '\\item  selection range: %s\r\n' : selection_range,
            '\\item  target count: %s\r\n' : target_count,
            '\\item  laser threshold: %s\r\n' : laser_threshold
        }
        FA.create_subsequent_latex(parameter_dict_latex,idx,header_latex,type=2)
    
# ******************************************************************************
# ************************* Main ***********************************************
# ******************************************************************************

    def feature_extraction(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--mode',type=int,
                            help='autonomous [0] or normal mode [1]',
                            default=1)
        parser.add_argument('-tr', '--total-range',type=int,
                            help='total range of lasers used',
                            default=200)
        parser.add_argument('-s','--sections',type=int,
                            help='amount of sections total range divided by',
                            default=5)
        parser.add_argument('-lc','--laser-count',type=int,
                            help='total amount of lasers. Robot specific',
                            default=666)
        parser.add_argument('-sr','--selection-range',type=int,
                            help='method used to relate positional values',
                            default=1)
        parser.add_argument('-tc','--target-count',type=int,
                            help='amount of positions in dm of which to relate',
                            default=35)
        parser.add_argument('-lt','--laser-threshold',type=float,
                            help='distance where objects within are relevant',
                            default=1.4)                          
        args = parser.parse_args()
        
        mode = args.mode

        total_range = args.total_range
        sections = args.sections
        laser_count = args.laser_count
        range_arr = self.get_ranges(total_range,sections,laser_count)

        selection_range = args.selection_range
        self.set_selection_range(selection_range)  

        target_count = args.target_count
        self.set_target_positions_arr(target_count)

        laser_threshold = args.laser_threshold
        self.set_laser_threshold(laser_threshold)       

        FA = FileAdministration()
        idx = FA.get_pipeline_index(current = True)
        bagfile_dir = FA.get_pipeline_directory(1,idx)
        dataset_dir = FA.get_pipeline_directory(2,idx)
        files = FA.get_rosbag_file(bagfile_dir)
        h5f = FA.create_h5py_file(dataset_dir,idx)

        self.safe_param(sections,laser_count,range_arr,
                        selection_range,target_count,laser_threshold,FA,idx)

        counter = 1
        
        for index, file in enumerate(files):
            print('Found ' + bagfile_dir + file + '.bag')
            print('Creating Bag: ' + str(counter) + ' from: '+ str(len(files)))

            dfs = self.get_df_prototype(rosbag.Bag(bagfile_dir\
                                                   + file\
                                                   + '.bag'), 
                                        self.get_dictionary(range_arr))

            df = self.get_df_syncronized(dfs)
            df = self.get_df_with_groundTruth(df)

            l = len(df)

            df.fillna(-1.0, inplace=True)
            df['camera'] = df['camera'].apply(lambda x: (x - x.mean()) 
                                                         / x.std())

            Xs = h5f.create_dataset('bag' + str(index) + '_x', 
                                    shape=(l, 64, 80, 3), 
                                    maxshape=(None, 64, 80, 3), 
                                    dtype=np.float, data=None, 
                                    chunks=True)
            Ys = h5f.create_dataset('bag' + str(index) + '_y', 
                                    shape=(l, 5 
                                    * len(self.target_positions_arr)), 
                                    maxshape=(None, 5 
                                    * len(self.target_positions_arr)), 
                                    dtype=np.float, data=None, chunks=True)
            
            cols = ['t_%.1f_%s' % (d, sensor) 
            for d in self.target_positions_arr                      
            for sensor in ['r1', 
                            'r2', 
                            'r3', 
                            'r4', 
                            'r5']]

            Xs[:] = np.stack(df['camera'].values)
            Ys[:] = df[cols].values
            counter += 1
        h5f.close()
    
if __name__ == '__main__':
    f = FeatureExtraction()
    f.feature_extraction()

