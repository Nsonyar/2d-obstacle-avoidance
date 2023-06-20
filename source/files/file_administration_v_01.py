import os
import re
import sys
import h5py
import shutil
import getopt

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
# ******************************************************************************
# ******************* Deprecated - replaced by v_02 ****************************
# ******************************************************************************
# ******************************************************************************

class FileAdministration:
    ''' Class responsible for the administration of all different types of files
        created in this project. This class provides functions for the creation
        of directories, functions to prove their validity and setting up
        conventions including raising error messages in case they are not met.
        This class was mainly used for development and testing and is replaced
        by file_administration_v02.py.
    '''

    def __init__(self, bag_dir_root=None, ds_dir_root=None, 
                    model_dir_root=None, v_bag_dir_root=None, 
                    v_model_dir_root=None, plot_dir_root = None,
                    loss_dir_root = None, result_dir_root = None):
        ''' Init function for the class FileAdministration initializing 
            variables used for the current instances once invoked
        Args:
            bag_dir_root:     root folder of all bag files
            ds_dir_root:      root folder of all dataset files
            model_dir_root:   root folder of all model files
            v_bag_dir_root:   root folder of all visualized bag videos
            v_model_dir_root: root folder of all visualized model videos
            plot_dir_root:    root folder of all plot files
            result_dir_root:  root folder of all result documentations
        '''

        self.bag_dir_root = bag_dir_root
        self.ds_dir_root = ds_dir_root
        self.model_dir_root = model_dir_root
        self.v_bag_dir_root = v_bag_dir_root
        self.v_model_dir_root = v_model_dir_root
        self.plot_dir_root = plot_dir_root
        self.loss_dir_root = loss_dir_root
        self.result_dir_root = result_dir_root

# ******************************************************************************
# ************************* Admin Functions ************************************
# ******************************************************************************

    def feature_extraction_file_admin(self,mode):
        ''' Administrator function responsible to call and provide all
            necessary steps to set up Feature extraction properly.
        Args:
            mode:   [0] autonomous mode, [1] developer mode 
        
        Return:
            files:      list of bagfiles to create dataset from
            h5f:        prototype of h5f dataset file
            bag_dir:    directory to bagfiles
            ds_idx:     pipeline index
        '''
        
        bag_dir, bag_idx =  self.get_bag_or_ds_dir(fileType = 0)
        ds_dir, ds_idx =  self.get_bag_or_ds_dir(fileType = 1)

        self.validate_idx(bag_idx, ds_idx)

        alternative_path = self.select_recording_location(mode)

        self.create_bag_dir(bag_dir, bag_idx, alternative_path)

        self.create_directory(ds_dir = ds_dir)

        files = self.get_rosbag_file(bag_dir)

        h5f = self.create_h5py_file(ds_dir,ds_idx)

        return files,h5f,bag_dir,ds_idx

    def training_model_file_admin(self,mode):
        ''' Administrator function, responsible to call and provide all
            necessary steps to set up the Training properly.
        Args:
            mode:   [0] autonomous mode, [1] developer mode 

        Return:
            idx:        pipeline index
            ds_dir:     directory of dataset
            model_dir:  directory of model
            loss_dir:   direcotry of loss graph
            file_count: No. of bagfiles for a specific pipeline index
        '''
        idx, dir_array = self.select_ds(mode)

        ds_dir = self.validate_file(idx,dir_array,fileType = 0)

        model_dir = self.create_model_dir(idx)

        loss_dir = self.get_loss_dir(idx)

        file_count = self.get_file_count('../files/bagfiles/', idx)

        return idx,ds_dir,model_dir,loss_dir,file_count

    def testing_model_file_admin(self,mode):
        ''' Administrator function, responsible to call and provide all
            necessary steps to set up the Testing properly.
        Args:
            mode:   [0] autonomous mode, [1] developer mode 
        Return:
            chosen_model:       path to model
            chosen_dataset:     path to dataset
            plot_dir:           directory of plots
            file_count:         No. of bagfiles for a specific pipeline index
        '''
        idx, dir_array = self.select_model(mode)
        chosen_model = self.validate_file(idx,dir_array, fileType = 1)
        chosen_dataset = self.get_chosen_ds(idx)
        plot_dir = self.get_plot_dir(idx)
        file_count = self.get_file_count('../files/bagfiles/', idx)
        return chosen_model,chosen_dataset,plot_dir,file_count

    def visualizing_model_admin(self):
        ''' Administrator function, responsible to call and provide all
            necessary steps to visualize the model.
        Return:
            idx:                pipeline index
            dir_array:          directory to model
            chosen_model:       path to model
            chosen_dataset:     path to dataset
            video_file_name:    path to video
        '''
        idx, dir_array = self.select_model(mode)
        chosen_model = self.validate_file(idx,dir_array, fileType = 1)
        chosen_dataset = self.get_chosen_ds(idx)
        video_file_name = self.get_video_file_name(idx,bag_index)

        return idx,dir_array,chosen_model,chosen_dataset,video_file_name

    def visualizing_dataset_admin(self):
        ''' Administrator function, responsible to call and provide all
            necessary steps to visualize the dataset.
        Return:
            idx:                pipeline index
            dir_array:          directory to model
            chosen_dataset:     path to dataset
            video_dir:          directory to video
        '''
        idx, dir_array = self.select_ds(mode)
        chosen_dataset = self.validate_file(idx,dir_array, fileType = 0)
        video_dir = self.get_video_dir(idx)
        file = chosen_dataset
        return idx,dir_array,chosen_dataset,video_dir,file

# ******************************************************************************
# ********************* Validation functions ***********************************
# ******************************************************************************

    def validate_idx(self, bag_idx, ds_idx):
        ''' Function will check whether the indexes of the bagfiles batch meets
            the created index of the dataset. Once this script is started, there
            should be the same amount of batches/folders of datasets and bagfile
            recordings.
        '''
        if(bag_idx != ds_idx):
            raise CustomError('Indexes not syncronized ! Check folders for '
                                'coherence !')

    def validate_file(self,idx,dir_array,fileType = 1):
        ''' Function applies several subfunctions in order to check whether
            the chosen dataset, model or recording is unique, is properly named 
            and stored and if the selection is valid in general.

        Args:
            idx:		folder index selected by the user (int)
            dir_array:	array containing the folder names with current datasets
                        available
            fileType:   0 = Dataset, 1 = Model, 2 = Recording
        Return:
            chosen_dataset: path to dataset with its name
        '''
        fullpath = os.path.join

        self.validate_selection(idx,dir_array,fileType)

        if(fileType == 0):
            directory = self.ds_dir_root + dir_array[idx-1]
        elif(fileType == 1):
            real_index = [i for i, s in enumerate(dir_array) if str(idx) in s]
            directory = self.model_dir_root + dir_array[real_index[0]]
        elif(fileType == 2):   
            return self.bag_dir_root + dir_array[idx-1]
        else:
            raise CustomError("Invalid fileType chosen ! Check manually !")

        file_array = os.listdir(directory)
        
        self.validate_folder(file_array)
        self.validate_file_index(idx,file_array)
                
        return fullpath(directory,file_array[0])

    def validate_selection(self,idx,dir_array,fileType):
        ''' In case of a dataset, function checks whether the selection of the 
            user is in the range of available dataset folders.

            In case of a model, function looking if a file corresponding to the 
            selection of the user, represented by idx, is available in the 
            list dir_array.
        
        Args:
            idx:        user selection (int)
            dir_array:  list containing the available subfolders
            fileType:   0 = Dataset, 1 = Model, 2 = recording
        '''
        if(fileType == 0):
            if (not(1<=idx<=len(dir_array))):
                raise CustomError('Invalid dataset index ! Please properly '
                                  'select dataset !')
        elif(fileType == 1):
            real_index = [i for i, s in enumerate(dir_array) if str(idx) in s]
            if (not(1<=len(real_index)<2)):
                raise CustomError('Invalid model index ! Please properly '
                                  'select model !')
        elif(fileType == 2):
            if (not(1<=idx<=len(dir_array))):
                raise CustomError('Invalid recording index ! Please properly '
                                  'select recording !')    
        else:
            raise CustomError('Invalid fileType given !')


    def validate_folder(self,file_array):
        ''' Function checks whether the folder chosen contains the allowed
            amount of 1 dataset per folder. If it does, the function passes.
            If it does not, the program exits.
        
        Args:
            file_array: array which contains the names of the files of the
                        chosen folder
        '''
        if(len(file_array)==1):
            return True
        elif len(file_array) == 0:
            raise CustomError('No file available !')
        else:
            raise CustomError('Invalid amount of files !')

    def validate_file_index(self,idx,file_array):
        ''' Function checks whether the dataset file name of the chosen folder 
            is matching the index chosen.
        
        Args:
            idx:        user selection (int)        
            file_array: array which contains the names of the files of the
                        chosen folder
        '''
        name = file_array[0]
        index_string = 'idx['+ str(idx) +']'
        if index_string in name:
            return True
        else:
            raise CustomError('Selected folder contains file with wrong index '
                              'set. Check conventions for dataset filenames !')

# ******************************************************************************
# ********************* Parameter processing ***********************************
# ******************************************************************************

    def process_arguments(self,argv):
        ''' C Style processing arguments mechanism. Currently not used.
        '''
        if(len(argv) == 0):
            return 1
        try:
            opts, args = getopt.getopt(argv,"m:")
            if(len(opts) != 1):
                raise CustomError('Wrong parameter set. Please launch as '
                                  'python3 feature_extraction -m <mode> !')
        except getopt.GetoptError:
            raise CustomError('python3 feature_extraction.py -m <mode>')
        for opt, arg in opts:
            if opt in ("-m"):
                try:
                    arg0 = int(arg)
                    if(arg0 == 0 or arg0 == 1): 
                        print('Parameter valid !')
                    else:
                        raise ValueError()
                except:
                    raise CustomError('Wrong data type as arg0 set !')
        return arg0

# ******************************************************************************
# ************************** User selection ************************************
# ******************************************************************************

    def select_ds(self,mode):
        ''' For the developer mode [1], function requesting the user to chose a 
            dataset for training, displaying all the available sets in a list to 
            chose from. For autonomous mode [1] the last created dataset is
            chosen to train from.
        
        Return:
            dir_array:	array containing the folder names of available datasets
            idx:        selection made by the user
            mode:       [0] autonomous mode, [1] developer mode
        '''
        if(os.path.exists(self.ds_dir_root)):

            dir_array = os.listdir(self.ds_dir_root)
            dir_array.sort(key = lambda x:int(x.split('_')[1]))
            print('Found datasets:')
            for i in range(len(dir_array)):
                if(i+1 == int(dir_array[i].split('_')[1])):
                    print('\t', i+1, ': ', dir_array[i])
                else:
                    raise CustomError('Incoherent index to dataset index !')

            if(mode == 1):
                idx = input('Please insert the dataset index: ')
                return int(idx), dir_array
            elif(mode == 0):
                idx = len(dir_array)
                return idx, dir_array
            else:
                raise CustomError('Wrong mode selected !')

        else:
            raise CustomError('No dataset folder and no dataset available !')

    def select_model(self,mode):
        ''' For the developer mode [1], function requesting the user to chose a 
            model for testing, displaying all the available models in a list to 
            chose from. For autonomous mode [1] the last created model is
            chosen to calculate from.
        
        Return:
            dir_array:	array containing the folder names of available models
            idx:        selection made by the user (int)
            mode:       [0] autonomous mode, [1] developer mode
        '''
        if(os.path.exists(self.model_dir_root)):
            dir_array = os.listdir(self.model_dir_root)
            dir_array.sort(key = lambda x:int(x.split('_')[1]))

            print('Found models:')
            for i in range(len(dir_array)):
                print('\t', int(dir_array[i].split('_')[1]), ': ', dir_array[i])

            if(mode == 1):
                idx = input('Please insert the model index: ')
                return int(idx), dir_array
            elif(mode == 0):
                idx = len(dir_array)
                return idx, dir_array
            else:
                raise CustomError('Wrong mode selected !')
        else:
            raise CustomError('No model folder available !')    

    def select_recording(self):
        ''' Function requesting the user to chose a recording for testing,
            displaying all the available recordings in a list to chose from.
        
        Return:
            dir_array:	array containing the folder names of available 
                        recordings
            idx:        selection made by the user (int)
        '''
        if(os.path.exists(self.bag_dir_root)):
            dir_array = os.listdir(self.bag_dir_root)
            dir_array.sort(key = lambda x:int(x.split('_')[1]))

            print('Found recordings:')
            for i in range(len(dir_array)):
                print('\t', int(dir_array[i].split('_')[1]), ': ', dir_array[i])

            idx = input('Please insert the recording index: ')
            return int(idx), dir_array
        else:
            raise CustomError('No recording folder available !')

    def select_recording_location(self,mode):
        ''' Function processes arguments from the command line first and
            then, depending on the result, checks if autonomous or developer 
            mode has been selected.
        Args:
            mode:   [0] autonomous mode, [1] developer mode
        '''
        if(mode == 1):
            inputVal = input('Use existing recording [0], Use new recording [1]')
            if(inputVal == '0'):
                idx, dir_array = self.select_recording()
                return self.validate_file(idx,dir_array, fileType = 2)
            elif(inputVal == '1'):
                return None
            else:
                raise CustomError('Invalid input. Please chose [0] to use '
                                  'existing or [1] to use new recording')
        elif(mode == 0):
            return None
        else:
            raise CustomError('Wrong mode set ! Please select 0 or 1')

# ******************************************************************************
# ********************* Directory or File obtainment ***************************
# ******************************************************************************

    def get_file_count(self, root_path, idx):
        ''' Function returning the file count of a given folder

        Args:
            root_path: root path of following format: ../files/<category>/             
            idx:       ID of folder to select for count (int)           
        '''   
        dir_string = os.listdir(root_path)
        if(len(dir_string) >= 1):
            path = root_path + dir_string[0].split('_')[0]+ '_' + str(idx)
            return len(os.listdir(path))

    def get_bag_or_ds_dir(self, fileType = 0):
        ''' Function returns consecutively numbered string, representing the
            path of where a bagfile or dataset will be stored.
        
        Args:
            fileType:   can be set to 0 or 1 where the former stands for bag and
                        the latter for a dataset directory string to be
                        returned.
        Return:
            Path representation as a string
            Index representing consecutive number of batch
        '''
        if(fileType == 0):
            file_dir = self.bag_dir_root
            partial_str = 'recording_'
            first_file_dir = self.bag_dir_root + 'recording_1/'
        elif(fileType == 1):
            file_dir = self.ds_dir_root
            partial_str = 'dataset_'
            first_file_dir = self.ds_dir_root + 'dataset_1/'
        else:
            raise CustomError("Wrong file type selected. Please chose 0 or 1 !")

        if(os.path.isdir(first_file_dir)):
            dir_array = os.listdir(file_dir)
            dir_array.sort(key = lambda x:int(x.split('_')[1]))   
            idx = int(dir_array[-1].split('_')[1])+1
            return  file_dir + partial_str + str(idx) + '/', idx
        else:
            return first_file_dir, 1

    def get_rosbag_file(self,bag_dir):
        ''' Function returning bagfiles from a given directory
        
        Args:
            bag_dir: directory of where bagfiles are stored
        Return:
            list of files
        '''
        return [file[:-4] 
                for file in os.listdir(bag_dir) 
                if file[-4:] == '.bag']

    def get_chosen_ds(self,idx):
        ''' Function returns a dataset path given an index as parameter. This 
            function is needed when testing a model by the Test() class in order
            to correctly correspond model and dataset. To avoid erroneous
            datasets to be selected, the folder where the set is stored as well
            as the file itself are going to be validated.

        Args:
            idx: number which indicated which dataset needs to be loaded (int)

        Return:
            path to dataset file with its corrensponding name
        '''
        fullpath = os.path.join
        dataset_dir = self.ds_dir_root + 'dataset_' + str(idx) + '/'
        file_array = os.listdir(dataset_dir)

        self.validate_folder(file_array)
        self.validate_file_index(idx,file_array)

        return fullpath(dataset_dir,file_array[0])

    def get_video_file_name(self,idx,bag_index):
        ''' Function returning the name and the path to a file used to store a
            video. If the directory to the model does not exist, it will be
            created. Otherwise it will be checked additionally, whether there
            is already a file stored with the selected bag_index or not. A
            custom error is raised, if it is attempted to overwrite existing
            files.
        Args:
            idx:        consecutive directory number this function call 
                        corresponds to (int)
            bag_index:  chosen number which bag is used to create the video from
        Return:
            path to file used to store a video
        '''
        v_model_dir = self.v_model_dir_root + 'video_' + str(idx) + '/'
        file_name_prototype = 'video_bag_' \
                              + str(bag_index) \
                              + '_idx[' \
                              + str(idx) \
                              + '].avi'
        if(not(os.path.isdir(v_model_dir))):
            self.create_directory(v_model_dir = v_model_dir)
            return v_model_dir + file_name_prototype
        else:
            file_array = os.listdir(v_model_dir)
         
            if(file_name_prototype in file_array):
                raise CustomError('Video with selected bag index existing '
                                  'already ! Following folder needs to be '
                                  ' manually checked: ../files/video_',idx)
            else:
                return v_model_dir + file_name_prototype

    def get_video_dir(self,idx):
        ''' Function returning a directory where videos created by the
            VisualizingDataset class are stored. If the directory doesn't exist, 
            it will be created. Otherwise, a custom error is raised with the 
            hint that a video set already exists for the chosen dataset.
        Args:
            idx:    consecutive directory number this function call 
                    corresponds to (int)
        Return:
            path to directory of where the plot will be stored
        '''
        v_bag_dir = self.v_bag_dir_root + 'video_' + str(idx) + '/'
        if(not(os.path.isdir(v_bag_dir))):
            self.create_directory(v_bag_dir = v_bag_dir)
            return v_bag_dir
        else:
            raise CustomError('Video on selected dataset existing already ! '
                              'Following folder needs to be manually '
                              'checked: ../files/visualized_bags/video_',idx)

    def get_plot_dir(self,idx):
        ''' Function returning a directory where plots created by the Test
            class are stored. If the directory doesn't exist, it will be
            created. Otherwise, a custom error is raised with the hint that
            a plot already exists for the chosen model.
        Args:
            idx:    consecutive directory number this function call 
                    corresponds to (int)
        Return:
            path to directory of where the plot will be stored
        '''
        plot_dir = self.plot_dir_root + 'plot_' + str(idx) + '/'
        if(not(os.path.isdir(plot_dir))):
            self.create_directory(plot_dir = plot_dir)
            return plot_dir
        else:
            raise CustomError('Plot for selected model existing already ! '
                              'Following folder needs to be manually '
                              'checked: ../files/plots/plot_',idx)

    def get_loss_dir(self,idx):
        ''' Function returning a directory where a loss graph created by the
            Train class is stored. If the directory doesn't exist, it will be
            created. Otherwise, a custom error is raised with the hint that
            a loss graph already exists for the chosen dataset.
        Args:
            idx:    consecutive directory number this function call 
                    corresponds to
        Return:
            path to directory of where the loss graph will be stored
        '''
        loss_dir = self.loss_dir_root + 'loss_' + str(idx) + '/'
        if(not(os.path.isdir(loss_dir))):
            self.create_directory(loss_dir = loss_dir)
            return loss_dir
        else:
            raise CustomError('Loss graph for selected set existing already ! '
                              'Following folder needs to be manually '
                              'checked: ../files/loss/loss_',idx)

# ******************************************************************************
# ************************* Directory or file creation *************************
# ******************************************************************************

    def create_bag_dir(self,bag_dir,bag_idx, alternative_path = None):
        ''' Function creates a bag directory, naming it as calculated in
            self.get_bag_directory(). It then further moves recorded bagfiles
            from /root/.ros/bagfiles/ to this created folder. To prevent
            incoherence, every file will be named with the consecutive number
            the folder name was created.
        
        Args:
            bag_dir:        directory of where the bagfiles are stored
            bag_index:      unique number for bagfiles to be identified and
                            correspond them to datasets and models calculated
                            upon them
            alt:            path of alternative selection of recordings
        '''
        fullpath = os.path.join
        if(alternative_path == None):
            bagfiles_path = os.environ['HOME']+'/.ros/bagfiles'
        else:
            bagfiles_path = alternative_path

        if(os.path.isdir(bagfiles_path)):
            for dirname, dirnames, filenames in os.walk(bagfiles_path):
                if(len(filenames)==0):
                    raise CustomError('No bag files to archive !')
                self.create_directory(bag_dir = bag_dir)

                for fn in filenames:
                    source = fullpath(dirname, fn)
                    if fn.endswith('bag'):
                        if(alternative_path == None):
                            f = fn[:-4]\
                                                + '_idx['\
                                                + str(bag_idx)\
                                                + '].bag'
                        else:
                            f = fn.replace('['\
                                              + fn.split("[")[1].split("]")[0]\
                                              +']',
                                           '[' + str(bag_idx)+ ']')
                        shutil.copy(source, fullpath(bag_dir, f))
                break
        else:
            raise CustomError('Folder.ros/bagfiles or not recording available !'
                              ' Run data acquisition !')

    def create_h5py_file(self,ds_dir,ds_idx):
        return h5py.File(ds_dir+'ds_idx['+str(ds_idx)+'].h5', 'w')

    def create_model_dir(self,idx):
        ''' Function creating a model directory corresponding to the chosen
            dataset. If the directory already exist, the program is terminated 
            and the user recommended to chose another dataset.

        Args:
            idx: unique name of a model directory (int)

        Return:
            path of mode directory created
        '''
        if(os.path.isdir('../files/models/model_' + str(idx))==False):
            model_dir = '../files/models/model_' + str(idx) + '/'
            self.create_directory(model_dir = model_dir)
            return model_dir
        else:
            raise CustomError('Model folder for chosen dataset exist! '
                              'Chose a different dataset or delete folder '
                              'called model_',str(idx))

    def create_directory(self, bag_dir = None, ds_dir = None, model_dir = None,
                         plot_dir = None, v_bag_dir = None, v_model_dir = None,
                         loss_dir = None):
        ''' Function checks if given string for directory is meeting the set
            conventions, which can also be changed here if necessary. The string
            is checked against a regular expression and a directory is just
            created, if the string passes the query.
        Args:
            bag_dir:        string describing path to bagfiles
            ds_dir:         string describing path to datasets
            model_dir:      string describing path to models
            plot_dir:       string describing path to plots
            v_bag_dir:      string describing path to visualized bags
            v_model_dir:    string describing path to visualized models
            loss_dir        string describing path to a loss graph
        '''
        
        if(bag_dir):
            if(re.search('^..\/files\/bagfiles\/recording_([1-9][0-9]{0,2})\/$', 
                         bag_dir)):     
                os.makedirs(bag_dir)
            else:
                raise CustomError('Bag folder convention error. Please manually'
                                    ' check given string: ' + bag_dir)

        if(ds_dir):
            if(re.search('^..\/files\/datasets\/dataset_([1-9][0-9]{0,2})\/$', 
                         ds_dir)):
                os.makedirs(ds_dir)
            else:
                raise CustomError('Dataset folder convention error. Please '
                                  'manually check given string: ' + ds_dir)

        if(model_dir):
            if(re.search('^..\/files\/models\/model_([1-9][0-9]{0,2})\/$', 
                         model_dir)):
                os.makedirs(model_dir)
            else:
                raise CustomError('Model folder convention error. Please '
                                  'manually check given string: ' + model_dir)

        if(plot_dir):
            if(re.search('^..\/files\/plots\/plot_([1-9][0-9]{0,2})\/$', 
                         plot_dir)):
                os.makedirs(plot_dir)
            else:
                raise CustomError('Plot folder convention error. Please '
                                  'manually check given string: ' + plot_dir)

        if(v_bag_dir):
            if(re.search('^..\/files\/visualized_bags\/video_([1-9][0-9]{0,2})\/$', 
                         v_bag_dir)):
                os.makedirs(v_bag_dir)
            else:
                raise CustomError('Visualized bag folder convention error. '
                                  'Please manually check given string: ' 
                                  + v_bag_dir)

        if(v_model_dir):
            if(re.search('^..\/files\/visualized_models\/video_([1-9][0-9]{0,2})\/$', 
                         v_model_dir)):
                os.makedirs(v_model_dir)
            else:
                raise CustomError('Visualized model convention error. Please '
                                  'manually check given string: ' + v_model_dir)
                        
        if(loss_dir):
            if(re.search('^..\/files\/loss\/loss_([1-9][0-9]{0,2})\/$', 
                         loss_dir)):
                os.makedirs(loss_dir)
            else:
                raise CustomError('Loss graph convention error. Please '
                                  'manually check given string: ' + loss_dir)



# ******************************************************************************
# ************************* Markdown Functions *********************************
# ******************************************************************************
    
    def create_first_markdown(self,parameter_dict,header,idx = None):
        ''' Function used to create a new markdown file from a dictionary given.
            Function is always called by data aquisiton and additionally by
            any other part of the pipeline if a previous instance has not
            generated it yet.

        Args:
            parameter_dict: Text to generate the markdown from
            header:         Header for the markdown file
            idx:            pipeline index
        '''
        if(os.path.exists(self.result_dir_root)):
            print("Result folder exist and is not created !")
        else:
            os.makedirs(self.result_dir_root)
        if(idx == None):
            if(os.path.exists(self.ds_dir_root)):
                idx = len(os.listdir(self.ds_dir_root)) + 1
            else:
                print('Dataset folder not existent. Pipeline index set to 1 !')
                idx = 1
        if(os.path.exists(self.result_dir_root + 'results['+str(idx)+'].md')):
            raise CustomError('Markdown existing already. Manually check !')
        else:
            f= open(self.result_dir_root + 'results_idx['+str(idx)+'].md','w+')
            f.write('# Results Model_%d\r\n' %idx)
            f.write(header)
            for key, value in parameter_dict.items():
                f.write(key % value)
            f.close()

    def create_subsequent_markdown(self,parameter_dict,idx,header):
        ''' Function used to generate a subsequent markdown file. Function adds
            information to an existing markdown. If no markdown exist, a first
            markdown file is created.
        Args:
            parameter_dict: Text to generate the markdown from
            idx:            pipeline index
            header:         Header for the markdown file
        '''
        if(os.path.exists(self.result_dir_root+'results_idx['+str(idx)+'].md')):
            f= open(self.result_dir_root + 'results_idx['+str(idx)+'].md','a+')
            f.write(header)
            for key, value in parameter_dict.items():
                f.write(key % value)
            f.close()
        else:
            self.create_first_markdown(parameter_dict,header,idx=idx)