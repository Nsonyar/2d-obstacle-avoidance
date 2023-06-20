import os
import sys
import time
import shutil
from datetime import datetime
from source.files.file_administration_v_02 import FileAdministration


class CustomError(Exception):
	''' Custom Class to provide the user with a proper error message in case
		some exception is raised.
	
	Args:
		msg: 	optional parameter to provide a message to the user
		index: 	optional parameter to indicate further index information about
				specific files corresponding to the error raised
				
	Return: 	Error message
	'''
	def __init__(self, msg = None, index = None):
		if msg:
			self.message = msg
			self.index = index
		else:
			self.message = None

	def __str__(self):
		if self.message and not(self.index):
			return 'CustomError, {0}'.format(self.message)
		elif self.message and self.index:
			return 'CustomError, {0}{1} !'.format(self.message,self.index)
		else:
			return 'CustomError has been raised'

class LaunchPipeline:
    ''' Class launched from launch_pipeline.sh used to run the entire project
        autonomously.
    '''

    def script_location_setup(self):
        ''' Function responsible to create an array of scripts which need to be
            called to run the pipeline.  
        '''
        self.script_list = []

        #ROS Scripts
        roslaunch = 'roslaunch ss20_01_data_acquisition '
        launchfile = 'ss20_01_data_acquisition.launch '
        args = "args:='-m 0 -a 5'"
        ''' Arguments
            [-m <value>] Can be set to 0 or 1. Value 0 will launch the robot
            normally recording the amount of bagfiles stated in [-a <value>].
            If [-m <value>] is set to 1, the robot will  add bagfiles to the
            collection already existing in /root/.ros/bagfiles/
        '''
        #Python Scripts
        interpreter = 'python3 '
        location = '../'
        p_scripts = ['ss20_02_feature_extraction/feature_extraction.py',
                     'ss20_03_training_model/training_model.py',
                     'ss20_04_testing_model/testing_model.py',
                     'ss20_05_visualizing_model/visualizing_model.py',
                     'ss20_06_visualizing_dataset/visualizing_dataset.py']
        
        self.script_list.append(roslaunch + launchfile + args)
        for i in range(len(p_scripts)):
            self.script_list.append(interpreter + location + p_scripts[i])

    def launch_pipeline(self):
        ''' Function to launch the pipeline. The pipeline is divided in 6 stages
            from which it can be run from if a certain stage fails. If the
            pipeline run fails at stage 3 for example, it can be rerun directly 
            from stage 3 and skip the first 2 steps. The pipeline cannot be
            started for the first time from another than stage 1 as files
            created are passed from stage to stage and would not be available if
            launched from any stage but 1.
            
            Stage 1:
                Here the user can chose whether to record new files or use
                existing files from previous runs. If existing files are used,
                stage 1 finishes, after copying the amount of chosen files to
                a folder of where they can be used at stage 2.
        '''
        self.script_location_setup()

        input_stage = input('Launch pipeline at stage [1,2,3,4,5,6]')
        start = datetime.now()

        try:
            stage = int(input_stage)
        except:
            raise CustomError('Input_stage cannot be cast to int. Pleas enter '
                              'proper input ! ')

        if(stage == 1):
            amount = 0
            if(os.path.isdir('./1_bagfiles')):
                bag_directories = os.listdir('./1_bagfiles')
            
                for i in range(len(bag_directories)):
                    amount = amount + len(os.listdir('./1_bagfiles/' \
                                                    + bag_directories[i]))

                user_input = input('Record new files [0]. Define amount of files ' \
                              '(max: '+ str(amount) + ') to use: ')
            else:
                user_input = input('No files available. Please proceed with [0] to '\
                              'record new bagfiles !')
                if user_input not in {'0'}:
                    raise CustomError('Wrong input set. Please Manually check !')
            try:
                user_input = int(user_input)
            except:
                raise CustomError('Wrong input type. Try again !')
            if(user_input == 0):
                print('Stage 1 launched !')
                ret_1 = os.system(self.script_list[0])
                print(ret_1)
                stage = 2

            elif(4<user_input< amount):

                print('Stage 1 launched !')

                #File Administration setup
                FA = FileAdministration()
                FA.create_root_directories()
                idx = FA.get_pipeline_index()
                FA.create_pipeline_directories(idx)
                
                #Set up first markdown file
                images = user_input * 60
                header = "## Data acquisition \r\n"
                parameter_dict = {
                    '- recorded bagfiles: %s\r\n': user_input,
                    '- recorded images: %s\r\n' : images
                }
                FA.create_first_markdown(parameter_dict,header,idx)

                #Set up first latex file
                header_latex = '\\textbf{Data acquisition}\r\n'
                parameter_dict_latex = {
                    '\\item recorded bagfiles: %s\r\n': user_input,
                    '\\item recorded images: %s\r\n' : images
                }
                FA.create_first_latex(parameter_dict_latex,header_latex,idx)

                FA.reuse_bagfiles(bag_directories,user_input,idx)
                FA.rename_bagfiles(idx)

                stage = 2
            else:
                raise CustomError('Wrong amount of input set. Please check !')

        if(stage == 2):
            print('Stage 2 launched !')
            ret_2 = os.system(self.script_list[1])
            print(ret_2)
            if(ret_2 != 0):
                print("Pipeline failed at step 2 !")
                sys.exit(1)
            stage = 3

        if(stage == 3):
            print('Stage 3 launched !')
            ret_3 = os.system(self.script_list[2])
            print(ret_3)
            if(ret_3 != 0):
                print("Pipeline failed at step 3 !")
                sys.exit(1)
            stage = 4

        if(stage == 4):
            print('Stage 4 launched !')
            ret_4 = os.system(self.script_list[3])
            print(ret_4)
            if(ret_4 != 0):
                print("Pipeline failed at step 4 !")
                sys.exit(1)
            stage = 5  
       

        if(stage == 5):
            print('Stage 5 launched !')
            ret_5 = os.system(self.script_list[4])
            print(ret_5)
            if(ret_5 != 0):
                print("Pipeline failed at step 5 !")
                sys.exit(1)
            stage = 6

        if(stage == 6):
            print('Stage 6 launched !')
            ret_6 = os.system(self.script_list[5])
            print(ret_6)
            if(ret_6 != 0):
                print("Pipeline failed at step 6 !")
                sys.exit(1)

        if stage not in {1,2,3,4,5,6}:
            raise CustomError('Wrong input set. Please Manually check !')

        end = datetime.now()
        minutes_diff = (end - start).total_seconds() / 60.0

        print('The pipeline completed in {:.2f} minutes !'.format(minutes_diff))

if __name__ == '__main__':
    LP = LaunchPipeline()
    LP.launch_pipeline()
