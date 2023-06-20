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

class FileAdministration:
    ''' Class responsible for the folder and file set up and maintenance for
        all files created by the pipeline. The class creates folders, checks
        them for consistency and properly sets up their names, relating files
        to each with a pipeline index specific for each run. This file is a
        subsequent version from file_administration_v_01.py. The reason to write
        implement this class was to remove redunancy and make the pipeline 
        easier to maintain.
    '''
    def __init__(self,ros = False):

        ''' Initialization function setting up the root and prototypes of the
            pipeline direcories specific to the pipeline index.
        Args:
            ros:    optional parameter indicating whether the calling file is a
                    ros node or not. Ros nodes have a different working
                    directory.
        '''

        if(ros == True):
            cwd = '../catkin_ws/src/ss20_lanz_2d_obstacle_avoidance/source/'
        else:
            cwd = '../'

        self.set_root_directories(cwd)
        self.set_pipeline_directory_prototypes()

# ******************************************************************************
# ************************* Directory setup ************************************
# ******************************************************************************

    def set_root_directories(self,cwd):
        ''' Function creates an array of root directories of each file type
            created by the pipeline. The directory numbers are related to each
            pipeline stage.
        '''
        self.root_list = []
        self.root_list.append(cwd + 'files/1_bagfiles/')
        self.root_list.append(cwd + 'files/2_datasets/')
        self.root_list.append(cwd + 'files/3_models/')
        self.root_list.append(cwd + 'files/4_plots/')
        self.root_list.append(cwd + 'files/5_model_videos/')
        self.root_list.append(cwd + 'files/6_dataset_videos/')
        self.root_list.append(cwd + 'files/summary/')

    def set_pipeline_directory_prototypes(self):
        ''' Set up of pipeline prototype directories. These strings are used
            to create pipeline directories specific to certain run
        '''
        self.proto =  ['bagfiles_','datasets_','models_','plots_',
                       'model_videos_','dataset_videos_','summary_']

    def create_root_directories(self):
        ''' Function called when launching the first pipeline creating all
            necessary root folders.
        '''
        ret = all(os.path.exists(self.root_list[i]) == True \
                  for i in range(len(self.root_list)))
        if(ret == False):
            for i in range(len(self.root_list)):
                os.makedirs(self.root_list[i])
    
    def get_pipeline_index(self,current = False):
        ''' Function called at the beginning of each pipeline, getting the
            pipeline index used for the current run over all instances
        Args:
            opt:    Optional parameter indicating whether to return next or
                    current pipeline index.
        '''
        len_list = []
        for i in range(len(self.root_list)):
            len_list.append(len(os.listdir(self.root_list[i])))
        if(self.check_all_equal(len_list)):
            if(current == False):
                return len(os.listdir(self.root_list[0]))+1
            elif(current == True):
                return len(os.listdir(self.root_list[0]))
            else:
                raise CustomError('Wrong datatype at requesting pipeline '
                                  'index !')
        else:
            raise CustomError('Pipeline incoherent. Please manually check !')
        
    def check_all_equal(self,len_list):
        ''' Function checks if values in a  list are all the same and 
            returns True if they are.
        Args:
            len_list: list with values
        '''
        return len_list[1:] == len_list[:-1]
    

    def create_pipeline_directories(self,idx):
        ''' Function creating all pipeline directories specific to its
            corresponding index. Function is called in data acquisition once
            the ROS node is started.
        '''
        for i in range(len(self.root_list)):
            os.makedirs(self.root_list[i] + self.proto[i] + str(idx) + '/')

    def get_pipeline_directory(self,fileType,idx):
        ''' Returns a pipeline directory specific to its index
        Args:
            fileType:   Specifies the type of file
            idx:        pipeline index
        '''
        return self.root_list[fileType-1] \
               + self.proto[fileType-1] \
               + str(idx) \
               + '/'

# ******************************************************************************
# ************************* File creation **************************************
# ******************************************************************************

    def create_h5py_file(self,dataset_dir,idx):
        return h5py.File(dataset_dir + 'dataset_'+ str(idx)+'.h5', 'w')

# ******************************************************************************
# ************************* Markdown Functions *********************************
# ******************************************************************************
    
    def create_first_markdown(self,parameter_dict,header,idx):
        ''' Function used to generate a first markdown file. 
        Args:
            parameter_dict: Text to generate the markdown from
            idx:            pipeline index
            header:         Header for the markdown file
        '''
        f= open(self.root_list[6] + self.proto[6] \
                + str(idx) + '/summary.md','w+')
        f.write('# Summary Model_%d\r\n' %idx)
        f.write(header)
        for key, value in parameter_dict.items():
            f.write(key % value)
        f.close()

    def create_subsequent_markdown(self,parameter_dict,idx,header):
        ''' Function used to generate a subsequent markdown file. Function adds
            information to an existing markdown.
        Args:
            parameter_dict: Text to generate the markdown from
            idx:            pipeline index
            header:         Header for the markdown file
        '''
        f= open(self.root_list[6] + self.proto[6] \
                + str(idx) + '/summary.md','a+')
        f.write(header)
        for key, value in parameter_dict.items():
            f.write(key % value)
        f.close()

# ******************************************************************************
# ************************* Latex Functions ************************************
# ******************************************************************************
    
    def create_first_latex(self,parameter_dict,header,idx):
        ''' Function used to generate a first latex file. 
        Args:
            parameter_dict: Text to generate the latex from
            idx:            pipeline index
            header:         Header for the latex file
        '''
        f= open(self.root_list[6] + self.proto[6]\
                + str(idx) + '/summary.tex','w+')
       
        f.write('\\newpage\r\n')

        f.write('\\subsubsection{Model ' + str(idx) \
                + '\\label{model_' + str(idx) + '} }\r\n')

        f.write('\\begin{multicols}{2}')
        f.write(header)
        f.write('\\begin{itemize}\r\n')
        f.write('\\setlength\\itemsep{0.1em}\r\n')
        for key, value in parameter_dict.items():
            f.write(key % value)
        f.write('\\end{itemize}\r\n')
        f.close()

    def create_subsequent_latex(self,parameter_dict,idx,header,type=None):
        ''' Function used to generate a subsequent latex file. Function adds
            information to an existing latex.
        Args:
            parameter_dict: Text to generate the latex from
            idx:            pipeline index
            header:         Header for the latex file
            type:           Special instruction
        '''
        f= open(self.root_list[6] + self.proto[6] \
                + str(idx) + '/summary.tex','a+')
        f.write(header)
        f.write('\\begin{itemize}\r\n')
        f.write('\\setlength\\itemsep{0.1em}\r\n')
        for key, value in parameter_dict.items():
            f.write(key % value)
        if(type == 1):
            for i in range(8):
                f.write('\\newline\r\n')
        f.write('\\end{itemize}\r\n')
        if(type==2):
            f.write('\\columnbreak')
        f.close()
    
    def create_last_latex(self,idx):
        ''' Create last latex instance with images
        Args:
            idx:    pipeline index
        '''
        f= open(self.root_list[6] + self.proto[6] \
                + str(idx) + '/summary.tex','a+')

        f.write('\\end{multicols}')
        f.write('\\begin{figure}[H]%[htbp]\r\n')
        f.write('\\centering\r\n')
        f.write('\\includegraphics[width=8cm,height=6cm]{3_models/models_' \
                + str(idx) + '/graph_' + str(idx) + '.png}\r\n')
        f.write('\\hspace{0.2 cm}\r\n')
        f.write('\\includegraphics[width=6cm,height=6cm]{4_plots/plots_' \
                + str(idx)+'/AUC_'+str(idx)+'.png}\r\n')
        f.write('\\caption{The left graph shows the validation versus the '
                'training loss while the right graph shows the summary of '
                'the Area Under the Receiver Operating Characteristic curve ' 
                'for all ranges from $\\{r_{1}, ... ,r_{n}\\}$ as well for all '
                'intermediary positions (distances).}\r\n')
        f.write('\\label{auc_'+str(idx)+'}\r\n')
        f.write('\\end{figure}\r\n')

        f.close()

# ******************************************************************************
# **************************** File obtainment *********************************
# ******************************************************************************

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

    def get_file_count(self, pipeline_dir):
        ''' Function returning the file count of a given folder

        Args:
            pipeline_dir: path of pipeline directory to count the files from             
        '''   
        return len(os.listdir(pipeline_dir))

# ******************************************************************************
# ************************** User selection ************************************
# ******************************************************************************

    def select_model(self):
        ''' For the developer mode [1], function requesting the user to chose a 
            model for testing, displaying all the available models in a list to 
            chose from. For autonomous mode [1] the last created model is
            chosen to calculate from.
        
        Return:
            dir_array:	array containing the folder names of available models
            idx:        selection made by the user (int)
        '''
        if(os.path.exists(self.root_list[2])):
            dir_array = os.listdir(self.root_list[2])
            dir_array.sort(key = lambda x:int(x.split('_')[1]))

            print('Found models:')
            for i in range(len(dir_array)):
                print('\t', int(dir_array[i].split('_')[1]), ': ', dir_array[i])

            idx = input('Please insert the model index: ')
            
            model_list = list(range(1,len(dir_array)+1))
            if int(idx) not in model_list:
                raise CustomError('Wrong input set. Please properly select !')
            else:
                return int(idx)
        else:
            raise CustomError('No model folder available !')

# ******************************************************************************
# ************************** Bagfile Administration ****************************
# ******************************************************************************

    def rename_bagfiles(self,idx,path=None):
        ''' Renaming of files
        '''
        if(path == None):
            path='./1_bagfiles/bagfiles_' + str(idx) + '/'

        for count, filename in enumerate(os.listdir(path)):
            dst = '['+ str(count+1) +']_'+ str(idx) + '.bag'
            src = path + filename 
            dst = path + dst 
            os.rename(src, dst)

    def reuse_bagfiles(self,bag_directories,input_bag,idx):
        ''' Function copies a specific amount of bagfiles to the corresponding
            folder in relation to a given pipeline index.
        '''
        fullpath = os.path.join
        files_dest_path = './1_bagfiles/bagfiles_' + str(idx) + '/'
        counter = 0
        for x in range(len(bag_directories)):
            for dirname, dirnames, filenames in os.walk('./1_bagfiles/' \
                                                        + bag_directories[x]):
                for filename in filenames:
                    source = fullpath(dirname, filename)
                    if filename.endswith("bag"):
                        shutil.copy(source,fullpath(files_dest_path,filename))
                        counter = counter + 1
                        if(counter >= int(input_bag)):
                            return True
                break

    def copy_bagfiles(self,bag_dir,idx):
        ''' Function renames bagfiles from /.ros/bagfiles folder and then copies
            them to the proper pipeline directory
        Args:
            bag_dir:    pipeline directory
            idx:        pipeline index
        '''
        fullpath = os.path.join
        bagfiles_path = os.environ['HOME']+'/.ros/bagfiles/'

        #self.rename_bagfiles(idx,bagfiles_path)

        if(os.path.isdir(bagfiles_path)):
            for dirname, dirnames, filenames in os.walk(bagfiles_path):
                if(len(filenames)==0):
                    raise CustomError('No bag files to archive !')

                for fn in filenames:
                    source = fullpath(dirname, fn)
                    if fn.endswith('bag'):
                        shutil.copy(source, fullpath(bag_dir, fn))
                break
        else:
            raise CustomError('Folder.ros/bagfiles not available !'
                              ' Run data acquisition !')

if __name__ == '__main__':
    FileAdministrationNew(ros = False)

