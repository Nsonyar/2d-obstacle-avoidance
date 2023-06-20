#!/usr/bin/env python
# main imports
import os
import sys
import cv2
import math
import rospy
import rosbag
import getopt
import random
import shutil
import actionlib
import numpy as np
from random import randint
from PIL import Image as Image_1
from nav_msgs.msg import Odometry
from math import pow, atan2, sqrt
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Int32, String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R
from source.files.file_administration_v_02 import FileAdministration
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, Point
from sensor_msgs.msg import LaserScan, Image, CompressedImage, PointCloud2
from tf.transformations import euler_from_quaternion, quaternion_from_euler

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

class Acquisition:
    ''' This class is representing a data acquisition controller, which records
        data used to train a neural network. The goal of this class is to find
        objects for recording, matching some preconditions for distance and
        positions, related to itself. To avoid collisions, a simple algorithm
        which calculates the opposite catheter of the y axis which represents
        the distance of obstacles from a possible passage towards a goal, is 
        implemented in the function called safe passage analysis.The robot turns 
        till a potential obstacle with free passage is found and then records 
        data while moving in a straight line towards the obstacle.The recorded 
        data are positional parameters, laser data and compressed rgg images. 
        The data is automatically stored under /root/.ros/bagfiles.The class 
        also allows the user to choose whether a new recording is taken place or 
        if recordings are added to existing bagfiles. It is recommended to start 
        the script from the pipeline launcher, which incorporates it within the 
        project. Parameters can be set manually or through the pipeline launcher
        . The script is also connected to the File Administration class, which 
        properly administrates the entire pipeline's inputs and outputs.
    '''

    def __init__(self, argv):
        #Variables containing rgb image data
        self.width = 0.0
        self.height = 0.0
        self.data = []
        self.data_compressed = []
        self.msg_img_raw = Image
        self.msg_img_raw_compressed = Image
        self.compressed_img_msg = Image

        #Robot position variables of model_states
        self.theta = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0

        #Robot position variables of odom
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.odom_msg = Odometry

        #Robot laser variables
        self.laser = []
        self.intensities = []
        self.laser_scan_msg = LaserScan

        #Robot motor variables
        self.pub = Twist
        self.twist = Twist()
        
        #Rosbag variables
        self.bag = rosbag.bag.Bag
    
        #Set of frequency
        self.rate = rospy.Rate(10)
        
        #Initialization of subscribers
        self.initialize_img_raw()
        self.initialize_img_raw_compressed()
        self.initialize_model_states()
        self.initialize_scan_raw()
        self.initialize_odom()
        self.initialize_point_cloud()

        #Initialization of publishers
        self.initialize_cmd_vel()

        #Start of main loop
        self.main(argv)

# ******************************************************************************
# ************************* Initialization functions ***************************
# ******************************************************************************

    def initialize_scan_raw(self):
        rospy.Subscriber('/scan_raw',LaserScan,self.LaserScan_callback)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan_raw',LaserScan,timeout = 5)
            except:
                rospy.loginfo("LaserScan not available !")
    
    def initialize_model_states(self):
        rospy.Subscriber('/gazebo/model_states',
                         ModelStates ,
                         self.ModelStates_callback)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/gazebo/model_states',
                                              ModelStates,timeout = 5)
            except:
                rospy.loginfo("ModelStates not available !")

    def initialize_odom(self):
        rospy.Subscriber('/mobile_base_controller/odom',
                         Odometry ,
                         self.Odometry_callback)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/mobile_base_controller/odom',
                                              Odometry,
                                              timeout = 5)
            except:
                rospy.loginfo("Odom not available !")

    def initialize_cmd_vel(self):
        self.pub = rospy.Publisher('/mobile_base_controller/cmd_vel',
                                   Twist ,
                                   queue_size = 5)

    def initialize_img_raw(self):
        rospy.Subscriber('/xtion/rgb/image_raw', 
                         Image, 
                         self.Image_callback)
        rgb_data = None
        while rgb_data is None:
            try:
                rgb_data = rospy.wait_for_message('/xtion/rgb/image_raw',
                                                  Image,
                                                  timeout = 5)
            except:
                rospy.loginfo("RGB Raw Image not available !")

    def initialize_img_raw_compressed(self):
        rospy.Subscriber('/xtion/rgb/image_raw/compressed', 
                         CompressedImage, 
                         self.Compressed_image_callback)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/xtion/rgb/image_raw/compressed',
                                              Image,
                                              timeout = 5)
            except:
                rospy.loginfo("RGB Raw compressed Image not available !")
    
    def initialize_point_cloud(self):
        rospy.Subscriber('/xtion/depth_registered/points',PointCloud2,
                         self.Pointcloud_callback)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/xtion/depth_registered/points',
                                              PointCloud2,
                                              timeout = 5)
            except:
                rospy.loginfo('Pointcloud not available !')

# ******************************************************************************
# ************************* Callback functions *********************************
# ******************************************************************************

    def Odometry_callback(self,msg):
        self.odom_msg = msg
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.odom_theta = euler_from_quaternion([q.x, q.y,q.z,q.w])[2]

    def Image_callback(self,msg):
        self.msg_img_raw = msg
        self.height = msg.height
        self.width = msg.width
        self.data = msg.data
        
    def Compressed_image_callback(self,msg):
        self.compressed_img_msg = msg
        self.data_compressed = msg.data

    def ModelStates_callback(self,msg):
        self.robot_x = msg.pose[1].position.x
        self.robot_y = msg.pose[1].position.y
        q = msg.pose[1].orientation
        self.theta = euler_from_quaternion([q.x, q.y,q.z,q.w])[2]

        #Alternative calculation with scipy
        #r = R.from_quat([q.x,q.y,q.z,q.w])
        #r.as_euler('zyx', degrees=True)

    def LaserScan_callback(self,msg):
        self.laser_scan_msg = msg
        self.laser = msg.ranges
        self.intensities = msg.intensities

    def Pointcloud_callback(self,msg):
        self.point_cloud_msg = msg

# ******************************************************************************
# ************************* Debugging functions ********************************
# ******************************************************************************

    def rgbData_debug(self):
        ''' Debugging function used to print some details about a camera image
        '''
        rospy.loginfo("Length of image_raw height: %f",self.height)
        rospy.loginfo("Length of image_raw width: %f",self.width)
        rospy.loginfo("Length of image_raw data: %f",len(self.data))

    def rgbCompressed_debug(self):
        ''' Debugging function used to print the array length of a compressed 
            image
        '''
        rospy.loginfo("Length of image_raw/compressed data: %f",
                      len(self.data_compressed))

    def laserScan_debug(self):
        ''' Debugging function used to print out laser data
        '''
        rospy.loginfo("LaserScan length: %f",len(self.laser))
        rospy.loginfo("Laser index 0: %f",self.laser[0])
        rospy.loginfo("Laser index 665: %f",self.laser[665])

        rospy.loginfo("Laser intensities length: %f",len(self.intensities))
        rospy.loginfo("Laser intensity at 0: %f",self.intensities[0])
        rospy.loginfo("Laser intensity at 665: %f",self.intensities[665])
    
    def modelStates_debug(self):
        rospy.loginfo("Robots position x: %f, y: %f, Theta: %f",
                      self.robot_x,
                      self.robot_y,
                      self.theta)  

    def laserThetaArray_debug(self,array):
        ''' Debugging function used to print out the thetaLaser Array
        '''
        for i in range(666):
            rospy.loginfo("Index: %f Laser value: %f Angle value: %f",
                          i,
                          array[0][i],
                          array[1][i])    

    def display_image(self):
        """ Test Function used to display random image
        """ 
        w, h = 640,480
        data = np.zeros((h, w, 3), dtype=np.uint8)
        #red patch in upper left
        data[0:640, 0:480] = [255, 5, 0] 
        print(data.shape)
        img = Image_1.fromarray(data, 'RGB')
        img.save('my.png')
        #img.show()

    def get_point_cloud_coordinate(self):
        print(len(self.point_cloud_msg.data))
        print(type(self.point_cloud_msg))
        print(type(self.point_cloud_msg.data))

        print(self.point_cloud_msg.data[0])

        #Printing height and width
        print(type(self.point_cloud_msg.width))
        print(type(self.point_cloud_msg.height))
        print(self.point_cloud_msg.width)
        print(self.point_cloud_msg.height)
        
        #Printing msg fields
        print(type(self.point_cloud_msg.fields))
        print(len(self.point_cloud_msg.fields))
        print(self.point_cloud_msg.fields)

        height = int(self.point_cloud_msg.height / 2)
        middle_x = int(self.point_cloud_msg.width / 2)
        middle = self.read_depth(middle_x,height,self.point_cloud_msg)
        # print(middle)
        # print(type(middle))

        cloud_points = list(pc2.read_points(self.point_cloud_msg, 
                                            skip_nans=True, 
                                            field_names = ("x", "y", "z"),
                                            uvs=[[middle_x,height]]))
        print(cloud_points)

# ******************************************************************************
# ************************* Image related functions ****************************
# ******************************************************************************

    def jpeg_to_numpy(self, image, size=None, normalize=False):
        ''' Converts a jpeg image in a 2d numpy array of BGR pixels and resizes 
            it to the given size (if provided). The function can be called as 
            follows: self.jpeg_to_numpy(self.data_compressed,(80,64), True)

        Args:
            image:      a compressed BGR jpeg image.
            size:       a tuple containing width and height, or None for no 
                        resizing.
            normalize:  a boolean flag representing wether or not to normalize 
                        the image.

        Returns:
            the raw, resized image as a 2d numpy array of BGR pixels.
        
        print(len(image))
        '''
        print(len(image))
        print(type(image))
        compressed = np.fromstring(image, np.uint8)
        raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)

        if size:
            raw = cv2.resize(raw, size)

        if normalize:
            raw = (raw - raw.mean()) / raw.std()
        
        rospy.loginfo(len(raw))
        rospy.loginfo(type(raw))
        rospy.loginfo(raw.shape)
        print(type(raw))
        img = Image_1.fromarray(raw, 'RGB')
        img.save('my.png')
        

    def ros_to_cv(self):
        ''' Function used to convert Ros message of type Image to cv_image 
            stored in a three dimensional numpy array. Here optionally a raw or 
            a compressed version can be choosen. For debugging purposes, the 
            shape and type of the image is being printed in a seperate window.
        '''
        bridge = CvBridge()         
        try:
            #cv_img = self.bridge.imgmsg_to_cv2(self.msg_img_raw, "bgr8")
            cv_img = bridge.compressed_imgmsg_to_cv2(self.compressed_img_msg, 
                                                       "bgr8")
        except CvBridgeError as e:
            rospy.loginfo("test1")
            print(e)

        rospy.loginfo(cv_img.shape)
        rospy.loginfo(type(cv_img))

        img = Image_1.fromarray(cv_img, 'RGB')
        img.show()

        return cv_img
    
    def divide_laser(self,range_start,range_end):
        ''' Debugging function which simply returns 1 if an
            object is in the range smaller than 1 meter in front of the robot. 
            0 is returned otherwise.
        
        Args:
            range_start:    lower range index of considered laser
            range_end:      upper range index of considered laser
        '''
        ranges = self.laser[range_start:range_end]
        for i in range(len(ranges)):
            if ranges[i]<1:
                return 1
        return 0
        
# ******************************************************************************
# ****** Functions concerning the setup, naming and creating of bagfiles *******
# ******************************************************************************

    def process_arguments(self,argv):
        ''' Checking if a parameter was given when launching the file. Possible
            parameter at this point is 0 or 1. If no parameter was given,
            argv[0] is set to default. This causes the node to launch in 
            developer mode, with the user to chose whether he wants to proceed
            adding bagfiles to an existing batch or start a new set of bagfiles.
            If the parameter 0 was given, the autonomous mode is activated, with
            the robot starting to collect a new set of recordings. If parameter
            1 is given, normal mode is started as well.
        '''
        if(argv[0]== 'default'):
            return 1, 0

        try:
            opts, args = getopt.getopt(argv,"m:a:")
            if(len(opts) != 2):
                raise CustomError('Wrong parameter set. Please launch as: '
                                  'roslaunch <package> <launch-file> args:="-m '
                                  '0 -a <value>"')
        except getopt.GetoptError:
            raise CustomError('roslaunch <package> <launch-file> args:="-m 0 '
                              '-a <value>"')
        for opt, arg in opts:
            if opt in ("-m"):
                try:
                    arg0 = int(arg)
                    if(arg0 == 0 or arg0 == 1 or arg0 == 2): 
                        print('Parameter valid !')
                    else:
                        raise ValueError()
                except:
                    raise CustomError('Wrong data type as arg0 set !')
            elif opt in ("-a"):
                try:
                    arg1 = int(arg)
                except:
                    raise CustomError('Wrong data type as arg1 set !')
        return arg0, arg1

    def delete_previous_bagfiles(self):
        ''' Function deleting previous stored bagfiles in home/<user>/.ros
            and stored bagfiles of bagfiles folder in /.ros
        '''
        dir_name_ros = os.environ["HOME"]+"/.ros/"
        ros_dir = os.listdir(dir_name_ros)

        for item in ros_dir:
            if item.endswith(".bag"):
                os.remove(os.path.join(dir_name_ros, item))

        dir_name_bagfiles = os.environ["HOME"]+"/.ros/bagfiles"
        
        if(os.path.isdir(dir_name_bagfiles)):
            bagfiles_dir = os.listdir(dir_name_bagfiles)

            for item in bagfiles_dir:
                if item.endswith(".bag"):
                    os.remove(os.path.join(dir_name_bagfiles, item))
    
    def move_bagfiles_to_bagfiles_folder(self):
        ''' Function moving bagfiles from /home/<user>/.ros to bagfile folder
            in /.ros folder. This is done to make sure that only finished 
            bagfiles are later extracted from that folder and not unfinished 
            ones. If bagfile folder doesn't exist, it will be created.
            Note that the outer for-loop with os.walk(), is interrupted with 
            break, as no subfolders need to be taken in consideration
        '''
        fullpath = os.path.join
        home_ros_directory = os.environ["HOME"]+"/.ros/"
        bagfiles_directory = os.environ["HOME"]+"/.ros/bagfiles"

        if(os.path.isdir(bagfiles_directory)==False):
            os.mkdir(bagfiles_directory)

        for dirname, dirnames, filenames in os.walk(home_ros_directory):
            for filename in filenames:
                source = fullpath(dirname, filename)
                if filename.endswith("bag"):
                    shutil.move(source, fullpath(bagfiles_directory, filename))
            break

    def rosbag_setup(self,idx):
        ''' Function determining the name and type of bag files used to record 
            instances of current topics
        '''
        self.bag = rosbag.Bag('[1]_'+ str(idx) +'.bag','w')
    
    def next_bagfile(self,counter,idx):
        ''' Function renaming the bagfile after storage of the previous one in 
            home/.ros

        Args:
            Counter used to give the bag file different names
        '''
        name = '['+ str(counter) + ']_' + str(idx) + ".bag"
        rospy.loginfo("Rosbag file name: %s",name)
        self.bag = rosbag.Bag(name,'w')
        
    def record_instance(self):
        ''' Function writes instances of Odometry, LaserScan and 
            CompressedImage data and stores them as a 
            bag file whose name is defined in the function def 
            rosbag_setup(self).
        '''
        self.bag.write('odom',self.odom_msg)
        self.bag.write('laser',self.laser_scan_msg)
        self.bag.write('image',self.compressed_img_msg)
    
    def get_counter(self):
        ''' Used if previous bagfiles are not deleted.

        Return: subsequent value of last bagfile recorded
        '''
        return len(os.listdir(os.environ["HOME"]+"/.ros/bagfiles"))
    
    def set_bagfile_limit(self, arg1, d_limit):
        ''' Function checks whether arg1 is set to zero or above. If set to 0,
            a default limit for developer mode is set. If set > 0, arg1 will be 
            used as limit for maximum bagfiles to be recorded.
        Args:
            arg1:       defining the amount of bagfiles to be recorded. If no 
                        bagfiles are set, default limit is chosen
            d_limit:    default limit for bagfiles defined in set_parameters()
        '''
        if(arg1 == 0):
            return d_limit
        elif(arg1 > 0):
            return arg1
        else:
            raise CustomError('Wrong bagfile limit set. Please check arg1 !')


    def mode_selection(self,mode):
        ''' The following lines have two functionalities. If mode is 0, the
            autonomous mode is chosen. Existing bagfiles in .ros are deleted
            and the robot starts to record a new set of files. If mode is 1,
            the developer mode is chosen where the user can chose whether to 
            create a new bagfiles batch or to keep adding recordings to an 
            existing batch. If mode 2 is chosen, the node simply records files
            without creating any pipeline direcotry.
        Args:
            mode:   [0] autonomous mode, [1] developer mode [2] recording mode

        Return:     subsequent value of last bagfile recorded 
        '''

        if(mode == 0):
            self.delete_previous_bagfiles()
            return 0
        elif(mode == 1 or mode == 2):
            selection = input('New recording [0], Add to existing recording '
                              ' [1]')
                
            if(selection == "0"):
                self.delete_previous_bagfiles()
                return 0
            elif(selection == "1"):
                dir_name_ros = os.environ["HOME"]+"/.ros/"
                ros_dir = os.listdir(dir_name_ros)

                for item in ros_dir:
                    if item.endswith(".bag"):
                        os.remove(os.path.join(dir_name_ros, item))
                return self.get_counter()
        else:
            print("Wrong option chosen. Check process_arguments function !")

# ******************************************************************************
# ************************* Trajectory functions *******************************
# ******************************************************************************

    def create_theta_array(self):
        ''' Function used to create a numpy array with shape 2:666 populated 
            with angle values. Function is just called once

        Return: 
            Filled numpy array used to calculate robot coordinate system
        '''
        array = np.empty((2,len(self.laser)))

        angleIncrement = 0.00577401509508
        
        #Filling array with angle values. Range is 0:333. Angle 0 is at 333
        index = 0
        for i in range(333,-1,-1):
            array[1][i] = index * angleIncrement
            index+=1

        #Filling array with angle values. Range is 334:665. Angle 0 is at 333
        index = 1
        for i in range(334,666,1):
            array[1][i] = index * -angleIncrement
            index+=1

        return array

    def create_theta_laser_array(self,array):
        ''' Loop filling the theta array with laser data

        Args:
            array: Array filled with Theta values of length 666
        
        Return:
            Returns the filled array with shape of [2:666]
        '''
        for i in range(len(self.laser)):
            array[0][i] = self.laser[i]
        return array

    def calc_coordinate_in_robot_system(self,distance):
        ''' Function returning the goal coordinates in the robot coordinate
            system. This is used to calculate the start and goal position.
        Args:
            distance:   distance of the goal from the robots current position. 
                        The longitudinal axis of the robot is considere the y
                        coordinate.
        '''
        coordinate = []
        coordinate.append(0)
        coordinate.append(distance)
        rospy.loginfo("robot system x %f robot system y: %f",
                coordinate[0],
                coordinate[1])
        return coordinate

    def calc_coordinate_in_robot_system_old(self,index,array,hypotenuse):
        ''' Function deprecated. Currently not used.
            Function calculating x and y value, given the hypotenuse along with 
            an angle. The coordinates are in the robot coordinate system 
            considering the longitudinal axis of the robot as y

        Args:
            index:      Index of the theta-laser array whose respective 
                        coordinate needs to be calculated 
            array:      Theta-laser array filled with data of robot coordinate 
                        system
            hypotenuse: Length from robot of where to calculate coordinate from

        Return:
            Returns the coordinate calculated
        '''
        print("ANGLE")
       
        angle = array[1][index]
        print(angle)
        coordinate = []
        if(angle>0):
            coordinate.append(math.sin(angle)*hypotenuse)
            coordinate.append(math.cos(angle)*hypotenuse)
        else:
            coordinate.append(math.sin(angle)*hypotenuse)
            coordinate.append(math.cos(angle)*hypotenuse)

        rospy.loginfo("robot system x %f robot system y: %f",
                      coordinate[0],
                      coordinate[1])
        return coordinate
    
    def get_interm_goal_coordinates(self,coordinate,robot_x,robot_y,theta):
        ''' Function transfering coordinates from robot coordinate system to 
            world coordinate system with transformation matrix covering first 
            the rotation and then the translation part

        Args:
            Coordinate of robot operating system

        Return:
            Values in world coordinates
        '''
        if(theta < 0):
            thetaTemp = -1*(abs(theta) + math.pi/2)
        else:
            thetaTemp = theta - math.pi/2

        rospy.loginfo("thetaReal: %f",theta)
        rospy.loginfo("thetaTemp: %f", thetaTemp)

        #Rotation part   
        x_rot = coordinate[0] \
                * math.cos(thetaTemp) \
                - coordinate[1] \
                * math.sin(thetaTemp)
        y_rot = coordinate[0] \
                * math.sin(thetaTemp) \
                + coordinate[1] \
                * math.cos(thetaTemp)
        rospy.loginfo("x_rot: %f y_rot: %f",x_rot,y_rot)
        
        #Translation part
        rospy.loginfo("robot_x: %f robot_y: %f",robot_x,robot_y)
        x_new = robot_x + x_rot
        y_new = robot_y + y_rot

        rospy.loginfo("x_new: %f y_new: %f",x_new,y_new)
        return x_new, y_new

    def halt_robot(self):
        ''' Function to stop robot immediately
        '''
        for _ in range(100):
            self.twist.angular.z = 0.0
            self.twist.linear.x = 0.0
            self.twist.linear.y = 0.0
            self.twist.linear.z = 0.0
            self.twist.angular.x = 0.0
            self.twist.angular.y = 0.0
            self.pub.publish(self.twist)
    
    def find_next_object(self,laser, laser_theta_array, rec_dist, t_safe,
                         counter,rand):
        ''' Function checking if Laser values from 333 is pointing to 
            an object right in front of it. As a second condition, there needs
            to be a free passage from the robots current position, to a possible
            goal. This calculation is done in self.safe_passage_analysis. For 
            the entire function to return as True, the object additionally needs 
            to be more than 4 meter away and none of the mentioned lasers can 
            have inf as its value. If not all condition are matched, the robot
            will keep rotating.

        Args:
            laser:              Current laser values
            laser_theta_array:  Array filled with current laser values and their
                                corresponding angles in robot coordinate system
            rec_dist:           recording distance set set_parameters()

        Return:
            Boolean value if condition is True
        '''
        tmp = [laser[333]]

        r1 = all(i > (rec_dist + 0.5) and i != float('inf') for i in tmp)
        r2 = self.safe_passage_analysis(laser_theta_array,tmp[0]-1,t_safe)

        #rospy.loginfo("r1: %f",r1)
        #rospy.loginfo("r2: %f",r2)

        if r1 == True and r2 == True:
            self.halt_robot()
            return True
        else:
            self.twist.angular.z = -0.3 if rand == False  else 0.3
            self.pub.publish(self.twist)
    
    def safe_passage_analysis(self,array,distance, t_safe):
        ''' Function checks if a passage from the robot with a diameter of 2
            meters to the goal is overlapped by obstacles or not. This allows
            the robot to reach the goal without collisions. The function can
            calculate the opposite catheter as the hypotenuse and angle are
            given. The obosite catheter gives the distance from the y axis of 
            the robot coordinate system and an obstacle just observing values
            which are lower than the distance parameter.
        Args:
            array:      2 dim nd.array containing angles and laser data
            distance:   > (rec_dist + 0.5) meter -1
        '''
        for i in range(len(array[0][120:546])):
            if(array[0][i] != float('inf') and array[0][i] < distance):
                res = array[0][i] * abs(math.sin(array[1][i]))
                if(res < (t_safe / 2)):
                    return False
        return True

    
    def move_towards_goal(self,x,y,robot_x,robot_y,theta,twist,pub):
        ''' Function responsible for moving a robot towards a given goal.
            Currently not used (Deprecated). Replaced by 
            self.move_towards_goal_new()

        Args:
            x:          x coordinate of goal to move to
            y:          y coordinate of goal to move to
            robot_x:    x coordinate of current position
            robot_y:    y coordinate of current position
            theta:      current theta value of the robot
            twist:      message used to publish data to the motor
            pub:        publisher variable to publish twist to motor topic
        '''
        if self.angular_vel(x,y,robot_x,robot_y,theta) > math.pi:
            twist.angular.z = self.angular_vel(x,
                                               y,
                                               robot_x,
                                               robot_y,
                                               theta)  - 2 * math.pi
            twist.linear.x = 0.5
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            pub.publish(twist)

        elif self.angular_vel(x,
                              y,
                              robot_x,
                              robot_y,
                              theta) < math.pi and \
             self.angular_vel(x,
                              y,
                              robot_x,
                              robot_y,
                              theta)  > - math.pi:

            twist.angular.z = self.angular_vel(x,y,robot_x,robot_y,theta) 
            twist.linear.x = 0.5
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            pub.publish(twist)

        else:
            twist.angular.z = 2 \
                              * math.pi \
                              - self.angular_vel(x,
                                                 y,
                                                 robot_x,
                                                 robot_y,
                                                 theta) 
            twist.linear.x = 0.5
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            pub.publish(twist)

    def move_towards_goal_new(self,x,y,robot_x,robot_y,theta,twist,pub,state):
        ''' Function responsible for moving a robot towards a given goal. This
            function replaces the previous function used and provides a more
            robust implementation.
        Args:
            x:          x coordinate of goal to move to
            y:          y coordinate of goal to move to
            robot_x:    x coordinate of current position
            robot_y:    y coordinate of current position
            theta:      current theta value of the robot
            twist:      message used to publish data to the motor
            pub:        publisher variable to publish twist to motor topic
        '''
        twist.angular.z = self.angular_vel_new(x,y,robot_x,robot_y,theta)
        if(state == 2):
            twist.linear.x = 0.8
        else:
            twist.linear.x = 0.5
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        pub.publish(twist)

    def calc_atan2_new(self,y,robot_y,x,robot_x):
        ''' Mapping atan2 to range from 0 to 2pi
        Args:
            y:          goal coordinate
            robot_y:    robot coordinate
            x:          goal coordinate
            robot_x:    robot coordinate
        '''
        if(atan2(y - robot_y,x - robot_x) > 0):
            return atan2(y - robot_y,x - robot_x)
        else:
            return math.pi + atan2(y - robot_y,x - robot_x) + math.pi
    
    def calc_theta_new(self):
        ''' Mapping theta to range from 0 to 2pi
        '''
        if(self.theta>0):
            return self.theta
        else:
            return math.pi + self.theta + math.pi
    
    def angular_vel_new(self,x,y,robot_x,robot_y,theta):
        ''' Alternative function using different ranges for theta and atan2 with
            a mapping from 0 to 2pi. The advantage of this is to avoid a 
            flipping state when theta or atan2 are changing from pi to -pi.
        Args:
            x:          x goal coordinate
            y:          y goal coordinate
            robot_x:    x robot coordinate
            robot_y:    y robot coordinate
            theta:      angle in relation to world coordinates
        '''
        atan2_alt = self.calc_atan2_new(y,robot_y,x,robot_x)
        theta_alt = self.calc_theta_new()

        if(atan2_alt - theta_alt < -math.pi):
            return 2*math.pi - theta_alt + atan2_alt
        elif(atan2_alt - theta_alt > math.pi):
            return atan2_alt - 2*math.pi - theta_alt
        else:
            return atan2_alt - theta_alt

    def angular_vel(self,x,y,robot_x,robot_y,theta):
        ''' Function responsible to calculate the current angular velocity 
            needed to turn towards a specific direction

        Args:
            x: x coordinate of goal to turn to
            y: y coordinate of goal to turn to
            robot_x: x coordinate of current position
            robot_y: y coordinate of current position
            theta: current theta value of the robot
        
        Return:
            Angle towards given goal
        '''
        rospy.loginfo('Theta: %f',self.theta)
        rospy.loginfo('Atan2: %f',atan2(y - robot_y,x - robot_x))
        return 1 * (atan2(y - robot_y,x - robot_x) - theta)

    def distance(self,x1,y1,x2,y2):
        ''' Function used to calculate the euclidean distance between two given 
            points

        Args:
            x1: x coordinate of first point
            y2: y coordinate of first point
            x2: x coordinate of second point
            y2: y coordinate of second point

        Return:
            Euclidean distance
        '''
        xd = x1 - x2
        yd = y1 - y2
        return math.sqrt(xd*xd + yd*yd)

    def calculate_z(self,rand,g_angle,s_angle):
        ''' Calculating the angular velocity to rotate for a specific degree.
            This function avoids the robot overshooting and therfore preventing
            itself from spinning which can be a problem.
        Args:
            rand:       booelan variable responsible for the angular orientation
            g_angle:    goal angle which needs to be approached to
            s_angle:    start angle
        Return:
            Value which is fed directly to the robots angular motion sensor
        '''
        rospy.loginfo('theta_alt: %f',self.theta_alt(rand))
        if(g_angle > 2 * math.pi):
            if(s_angle < self.theta_alt(rand)+0.0001 <  2 * math.pi):
                return (g_angle - self.theta_alt(rand))*self.orientation(rand)
            else:
                return (g_angle-(2 * math.pi + self.theta_alt(rand))) \
                        * self.orientation(rand)
        else:
            return (g_angle - self.theta_alt(rand))*self.orientation(rand)

    def orientation(self,rand):
        ''' Depending on a boolean variable, the function either returns 1 or
            -1. This function is used to determin whether the robot should
            turn to the right or to the left.
        Args:
            rand: boolean variable
        Return:
            1 or -1
        '''
        if(rand == True):
            return 1
        elif(rand == False):
            return -1

    def theta_alt(self,rand):
        ''' Function used to calculate an alternative Theta value with ranges
            from 0 to 2pi or 2pi to 0 depending on the boolean variable.
        Args:
            rand: boolean variable
        Return:
            Theta displayed in the new range
        '''
        if(rand == True):
            if(self.theta < 0):
                return math.pi + self.theta + math.pi
            else:
                return self.theta
        elif(rand == False):
            if(self.theta > 0):
                return 2* math.pi - self.theta
            else:
                return abs(self.theta)
    
    def get_rand(self):
        return bool(random.getrandbits(1))

    def get_random_angle(self):
        return randint(45,90)

# ******************************************************************************
# ************************* Parameter Setup ************************************
# ******************************************************************************

    def set_parameters(self):
        ''' Function used to set parameters of the node
        '''
        ''' Getting a random boolean variable used to have the robot turning
            randomly either to the left or to the right
        '''
        rand = self.get_rand()
        ''' Variable used to indicate the current states of the robot. 
            In normal operation, states start with number 0
        '''
        state = 0
        s_state = 0
        t_state = 0

        ''' Variable representing the degree the robot turns away from obstacle
            after having reached the goal position. This is part of the 
            collision avoidance strategy.
        '''
        angle_deg = 45
        ''' Sets a time limit the robot tries to find an empty instance at state
            6
        '''
        t_limit = 20

        ''' Variable representing the recording distance.
        '''
        rec_dist = 3.5

        ''' Variable setting the safe passage threshold
        '''
        t_safe = 2

        ''' default limit for bagfiles to be recorded
        '''
        d_limit = 100

        safe_d = 0.35

        return rand,state,s_state,t_state,angle_deg,t_limit,rec_dist,t_safe,d_limit,safe_d

    def safe_param(self,bag_limit,t_limit,rec_dist,t_safe,FA,idx,safe_d):
        ''' Function sends the most important parameters to the File
            Administration class where a markdown and latex file are 
            automatically created. A further description of each parameter can 
            be seen in the dictionary sent.
        '''
        header = "## Data acquisition \r\n"
        parameter_dict = {
            '- recorded bagfiles: %s\r\n': bag_limit,
            '- recorded images: %s\r\n' : bag_limit*int(rec_dist)*20,
            '- safe distance: %s\r\n' : safe_d,
            '- timeout limit: %s\r\n' : t_limit,
            '- recording distance: %s\r\n' : rec_dist,
            '- safe passage diameter: %s\r\n' : t_safe
        }
        FA.create_first_markdown(parameter_dict,header,idx)

        header_latex = '\\textbf{Data acquisition}\r\n'
        parameter_dict_latex = {
            '\\item recorded bagfiles: %s\r\n': bag_limit,
            '\\item recorded images: %s\r\n' : bag_limit*60,
            '\\item angle avoidance: %s\r\n' : 'random int',
            '\\item timeout limit: %s\r\n' : t_limit,
            '\\item recording distance: %s\r\n' : rec_dist,
            '\\item safe passage diameter: %s\r\n' : t_safe
        }
        FA.create_first_latex(parameter_dict_latex,header_latex,idx)
        
# ******************************************************************************
# ************************* Point Cloud testing ********************************
# ******************************************************************************

    def read_depth(self,width, height, data):
        ''' Function to read out point cloud coordinate by giving it some height
            and width.
        Args:
            width:      x coordinate
            height:     y coordinate
            data:       point cloud data
        '''
        # read function
        if (height >= data.height) or (width >= data.width) :
            return -1
        data_out = pc2.read_points(data, field_names='y', skip_nans=False, 
                                   uvs=[[width,height]])
        int_data = next(data_out)
        #rospy.loginfo("int_data " + str(int_data))
        return int_data

# ******************************************************************************
# ************************* Main Loop ******************************************
# ******************************************************************************

    def main(self,argv):
        ''' Main function responsible for movement of the robot. The while loop 
            is set up as a state machine as follows:

            state 0:    Initial state after launching the node. Robot will turn 
                        till a specific amount of lasers (330:336) pointing 
                        directly on the object. It is testet furthermore, that 
                        the lasers are all above a specific threshold (4). Once 
                        the object is located, the robot will stop to turn and 
                        the state will change to 1.
            state 1:    At this state, a starting position, from where the 
                        recording is starting is going to be calculated. The 
                        robot now facing exactly towards the object. Here, 4 is 
                        being subtracted from the distance of the robot to the 
                        object and the result used as starting position for 
                        state 2.
            state 2:    At this state the robot is moving towards the starting 
                        position with a tolerance of 0.5. Once the starting 
                        position is reached, the robot is stopped and the state 
                        will change to 3.
            state 3:    At this state the robot is about 4 meters away and 
                        facing directly towards the object. A new coordinate, 
                        which is close to the object is calculated. Once the 
                        calculation is done, the state will change to 4.
            state 4:    At this state the robot will move towards the previous 
                        calculated coordinate, close to the object. The 
                        recording will be run till the coordinate close to the
                        object is reached. The file will be saved and the state 
                        will change to 5 with the robot trying to find a new 
                        object for the entire procedure to start again.
            state 5:    At this point the robot needs to turn away from the
                        obstacle before finding a new obstacle to approach to.
                        This has to be done to avoid collisions. The robot can
                        be made to turn for a specific amount of degrees. Once
                        the robot has turned correspondingly, the state is set
                        back to 0. It is important at this step to keep in mind
                        that the theta value once it reaches PI flips to minus
                        PI. The implemented functions avoid overshooting the
                        angular velocity passed to the publisher.
            state 6:    State 6 is just triggered every 3 recordings. At this 
                        state it is attempted to search an area without
                        obstacles in order to record a file without any input
                        from the camera except the empty floor and sky. This
                        state is necessary to have the prediction classifying
                        images without obstacles accordingly. After an empty
                        space is found, the robot will move 4 meters forward
                        from its current position. To achieve this, the state
                        from here is set to state 3 immediately. If the robot
                        does not find any area matching the set conditions for
                        a period of 20 seconds, state is set to 0 to avoid the
                        robot getting stuck in a loop.
        '''
        #Parameter setup
        rand,state,s_state,t_state,angle_deg,t_limit,rec_dist,t_safe,d_limit,safe_d \
        = self.set_parameters()
        
        #Debugging functions
        self.rgbData_debug()
        self.rgbCompressed_debug()
        self.modelStates_debug()
        self.laserScan_debug()

        #Check if argument was given
        arg0, arg1 = self.process_arguments(argv)

        #Sets the bag_limit if arg1 provided. Default value:= 100
        bag_limit = self.set_bagfile_limit(arg1,d_limit)

        #Default index     
        idx = 0

        if(arg0 != 2):

            #File Administration setup
            FA = FileAdministration(ros = True)
            FA.create_root_directories()
            idx = FA.get_pipeline_index()
            FA.create_pipeline_directories(idx)
            pipeline_dir = FA.get_pipeline_directory(1,idx)

            self.safe_param(bag_limit,t_limit,rec_dist,t_safe,FA,idx,safe_d)
     
        #Rosbag setup
        self.rosbag_setup(idx)

        #LaserTheta array setup
        theta_array = self.create_theta_array()

        #Request if new or existing batch used.
        counter = self.mode_selection(arg0)


        #start of infinite while loop with state machine
        while not rospy.is_shutdown():
            

            if(state == 0):
                rospy.loginfo("Robot at state: %f",state)
                laser_theta_array = self.create_theta_laser_array(theta_array) 
                if(self.find_next_object(self.laser, 
                                         laser_theta_array,
                                         rec_dist,
                                         t_safe,
                                         counter,
                                         rand)):
                    state = 1

            if(state == 1):
                rospy.loginfo("Robot at state: %f",state)
                dist = laser_theta_array[0][333] - (rec_dist + safe_d)
                rospy.loginfo("dist: %f",dist)

                robot_coordinate = \
                self.calc_coordinate_in_robot_system(dist)

                x_real, y_real = \
                self.get_interm_goal_coordinates(robot_coordinate,
                                                    self.robot_x,
                                                    self.robot_y,
                                                    self.theta)

                rospy.loginfo("x_real: %f y_real: %f",x_real,y_real)
                state = 2

            if(state == 2):
                rospy.loginfo("Robot at state: %f",state)
                if(self.distance(self.robot_x,
                                 self.robot_y,
                                 x_real,
                                 y_real) < safe_d):
                    self.halt_robot()
                    state = 3
                else:
                    self.move_towards_goal_new(x_real,y_real,self.robot_x,
                                               self.robot_y,
                                               self.theta,
                                               self.twist,
                                               self.pub,state)
            
            if(state == 3):
                rospy.loginfo("Robot at state: %f",state)
                laser_theta_array = \
                self.create_theta_laser_array(theta_array) 

                robot_coordinate = \
                self.calc_coordinate_in_robot_system(rec_dist)

                x_real, y_real = \
                self.get_interm_goal_coordinates(robot_coordinate,
                                                    self.robot_x,
                                                    self.robot_y,
                                                    self.theta)

                rospy.loginfo("x_real: %f y_real: %f",x_real,y_real)
                state = 4
            
            if(state == 4):
                rospy.loginfo("Robot at state: %f",state)
                if(self.distance(self.robot_x,
                                 self.robot_y,x_real,y_real) < safe_d):
                    self.halt_robot()
                    counter = counter + 1
                    self.bag.close()
                    self.move_bagfiles_to_bagfiles_folder()
                    rospy.loginfo("Counter in main loop: %f",counter)
                    self.next_bagfile(counter,idx)
                    if(counter > bag_limit):
                        #No copying is taking place if Mode 2 is selected
                        if(arg0 != 2):
                            FA.copy_bagfiles(pipeline_dir,idx)
                        rospy.loginfo("Bagfile limit of %f reached !", counter)
                        rospy.signal_shutdown('Node shutting down !')
                    #Every 4th round record empty instance
                    rand = self.get_rand()
                    if(counter % 7 == 0):
                        state = 6
                        t_state = 0
                    else:
                        state = 5
                else:
                    self.move_towards_goal_new(x_real,y_real,self.robot_x,
                                               self.robot_y,
                                               self.theta,
                                               self.twist,
                                               self.pub,state)
                    self.record_instance()
            
            if(state == 5):
                rospy.loginfo("Robot at state: %f",state)
                if(s_state == 0):
                    angle_deg = self.get_random_angle()
                    if(rand == False):
                        angle_deg * -1
                    angle_rad = angle_deg * math.pi / 180
                    g_angle = self.theta_alt(rand) + abs(angle_rad)
                    s_angle = self.theta_alt(rand)
                    s_state = 1
                    rospy.loginfo('angle_rad: %f', angle_rad)
                    rospy.loginfo('g_angle: %f',g_angle)
                    rospy.loginfo('s_angle: %f',s_angle)

                if(s_state == 1 ):
                    temp = self.calculate_z(rand,g_angle,s_angle)
                    rospy.loginfo('z: %f',temp)
                    rospy.loginfo('rand %f',rand)
                    if(round(temp,1) == 0.1 or round(temp,1) == -0.1):
                        state = 0
                        s_state = 0
                    else:
                        self.twist.angular.z = temp
                        self.pub.publish(self.twist)

            if(state == 6):
                rospy.loginfo("Robot at state: %f",state)
                now = rospy.get_rostime()
                if(t_state == 0):
                    start = rospy.get_rostime()
                    t_state = 1
                if(t_state == 1):
                    now = rospy.get_rostime()
                    dif = now.secs - start.secs
                    if(dif > t_limit):
                        state = 0
                if(all(ele > 13 for ele in self.laser[211:411])):
                    state = 3
                else:
                    self.twist.angular.z = 0.3 if counter % 2 == 0 else -0.3
                    self.pub.publish(self.twist)
            
            self.rate.sleep()

if __name__ == '__main__':
    
    rospy.init_node("data_acquisition")
    Acquisition(sys.argv[1:])
    rospy.spin()
