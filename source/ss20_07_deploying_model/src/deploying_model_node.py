#!/usr/bin/env python
# main imports
import os
import cv2
import sys
import math
import rospy
import rosbag
import shutil
import actionlib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as Image_1
from nav_msgs.msg import Odometry
from math import pow, atan2, sqrt
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Int32, String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import LaserScan, Image, CompressedImage
from source.ss20_03_training_model.training_model import Train
from source.files.file_administration_v_02 import FileAdministration
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, Point
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Deployment:
    ''' Node containing the deployment of a trained model. Consists of an
        obstacle avoidance controller with an implemented Bug Algorithm
        (currently not used) and an Intermediary Goal Algorithm. The node
        furthermore provides a heatmap which displays predictions made by the
        model, given an image. It further contains debugging and testing
        functions to manually measure the performance of the model.
    '''

    def __init__(self):
        #Robot position variables of model_states
        self.theta = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0

        #Intermediary goal variables
        self.int_goal_state = 0
        self.x_new = 0.0
        self.y_new = 0.0

        #Robot motor variables
        self.pub = Twist
        self.twist = Twist()

        #Variables containing rgb image data
        self.data_compressed = []

        #Set of frequency
        self.rate = rospy.Rate(5)

        #Initialization of subscribers
        self.initialize_img_raw_compressed()
        self.initialize_model_states()
        self.initialize_scan_raw()

        #Initialization of publishers
        self.initialize_cmd_vel()

        # Start of main loop function
        self.main()

# ******************************************************************************
# ************************* Initialization functions ***************************
# ******************************************************************************

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

    def initialize_cmd_vel(self):
        self.pub = rospy.Publisher('/mobile_base_controller/cmd_vel',
                                   Twist ,
                                   queue_size = 5)

# ******************************************************************************
# ************************* Debugging Functions ********************************
# ******************************************************************************
    def get_sys_path(self):
        print("SYSPATH")
        print(sys.path)
        print(os.path.abspath(os.getcwd()))
        print(os.path.dirname(os.path.abspath(__file__)))

    def print_coordinates(self,x,y,x_main,y_main):
        rospy.loginfo('x: %f',x)
        rospy.loginfo('y: %f',y)
        rospy.loginfo('x_main: %f',x_main)
        rospy.loginfo('y_main: %f',y_main)
        
# ******************************************************************************
# ************************* Callback Functions *********************************
# ******************************************************************************

    def Compressed_image_callback(self,msg):
        self.compressed_img_msg = msg
        self.data_compressed = msg.data

    def LaserScan_callback(self,msg):
        self.laser_scan_msg = msg
        self.laser = msg.ranges

    def ModelStates_callback(self,msg):
        self.robot_x = msg.pose[1].position.x
        self.robot_y = msg.pose[1].position.y
        q = msg.pose[1].orientation
        self.theta = euler_from_quaternion([q.x, q.y,q.z,q.w])[2]

# ******************************************************************************
# ************************* Image Functions ************************************
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
        '''
        
        compressed = np.fromstring(image, np.uint8)
        raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)

        if size:
            raw = cv2.resize(raw, size)

        if normalize:
            raw = (raw - raw.mean()) / raw.std()

        img = Image_1.fromarray(raw, 'RGB')
        img.show()

        return raw

# ******************************************************************************
# ************************* Heatmap Functions **********************************
# ******************************************************************************

    def numpy_to_heatmap_setup(self,n_dist):
        width = 5
        height = n_dist
        plt.ion()
        fig, ax = plt.subplots()

        array = np.zeros(shape=(height, width), dtype=np.uint8)
        array[0, 0] = 99
        #https://stackoverflow.com/questions/32633322
        #/changing-aspect-ratio-of-subplots-in-matplotlib
        axim = ax.imshow(array,aspect='auto')
        del array
        ax.set_xticks(range(5))
        #ax.set_xticklabels(range(5)[::-1])
        ranges = []
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        for i in range(5,0,-1):
            ranges.append('r'+ str(i).translate(SUB))
        ax.set_xticklabels(ranges)
        ax.set_yticks(range(n_dist))
        ax.set_yticklabels(range(n_dist)[::-1])
        return fig, axim
    
# ******************************************************************************
# ************************* Testing Functions **********************************
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

    def calculate_ground_truth(self, ranges):
        ''' Function used to test predictions for its accuracy. Function checks
            if obstacles interfer with ranges defined. If obstacles interfer,
            function calculates the mean distance over the partial range which
            is interfered by an obstacle.
        Args:
            ranges: Numpy array with following elements for example:
                    [233 273 274 314 315 355 356 396 397 437]
        '''
        r1 = [i < 4 for i in self.laser[ranges[0]:ranges[1]]]
        r2 = [i < 4 for i in self.laser[ranges[2]:ranges[3]]]
        r3 = [i < 4 for i in self.laser[ranges[4]:ranges[5]]]
        r4 = [i < 4 for i in self.laser[ranges[6]:ranges[7]]]
        r5 = [i < 4 for i in self.laser[ranges[8]:ranges[9]]]

        if True in r1:
            result_r1 = self.Average(self.laser[ranges[0]:ranges[1]])
        else:
            result_r1 = 0
        if True in r2:
            result_r2 = self.Average(self.laser[ranges[2]:ranges[3]])
        else:
            result_r2 = 0
        if True in r3:
            result_r3 = self.Average(self.laser[ranges[4]:ranges[5]])
        else:
            result_r3 = 0
        if True in r4:
            result_r4 = self.Average(self.laser[ranges[6]:ranges[7]])
        else:
            result_r4= 0
        if True in r5:
            result_r5 = self.Average(self.laser[ranges[8]:ranges[9]])
        else:
            result_r5 = 0

        rospy.loginfo('%f %f %f %f %f',result_r5,result_r4,result_r3,result_r2,
                      result_r1)


    def Average(self,lst):
        a = [x for x in lst if x != float('inf') and x < 4]
        return sum(a) / len(a)

# ******************************************************************************
# ************************* Trajectory Functions *******************************
# ******************************************************************************

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

    def move_towards_goal_alt(self,x,y,robot_x,robot_y,theta,twist,pub):
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
        twist.angular.z = self.angular_vel_alt(x,y,robot_x,robot_y,theta)
        twist.linear.x = 0.5
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        pub.publish(twist)
    
    def calc_atan2_alt(self,y,robot_y,x,robot_x):
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
    
    def calc_theta_alt(self):
        ''' Mapping theta to range from 0 to 2pi
        '''
        if(self.theta>0):
            return self.theta
        else:
            return math.pi + self.theta + math.pi
    
    def angular_vel_alt(self,x,y,robot_x,robot_y,theta):
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
        atan2_alt = self.calc_atan2_alt(y,robot_y,x,robot_x)
        theta_alt = self.calc_theta_alt()

        if(atan2_alt - theta_alt < -math.pi):
            return 2*math.pi - theta_alt + atan2_alt
        elif(atan2_alt - theta_alt > math.pi):
            return atan2_alt - 2*math.pi - theta_alt
        else:
            return atan2_alt - theta_alt

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

    def get_current_coordinate(self,counter,list_coordinates):
        ''' Providing coordinates given a counter
        Args:
            counter:            counter of coordinates
            list_coordinates:   list of coordinates
        '''
        return list_coordinates[counter][0], list_coordinates[counter][1] 

    def turn_left(self,twist,pub):
        ''' Function used to turn to the left (Bug Algorithm)
        '''
        twist.angular.z = 0.7
        twist.linear.x = 0.6
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        pub.publish(twist)

    def turn_right(self,twist,pub):
        ''' Function used to turn to the right (Bug Algorithm)
        '''
        twist.angular.z = -0.7
        twist.linear.x = 0.6
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        pub.publish(twist)

# ******************************************************************************
# ************************* Bug Algorithm **************************************
# ******************************************************************************
    
    def bug_algorithm(self,prediction,x_goal,y_goal):
        ''' Simple bug algorithm checking which part of the prediction contains
            less obstacles and then simply providing relevant angular velocity
            to the robots motor. Function is constantly called to check the
            presence of obstacles.
        Args:
            prediction: Models prediction
            x_goal:     x goal coordinate
            y_goal:     y goal coordinate
        '''
        left,mid,right = self.get_greenlight(prediction)
        if(left == False and mid == False and right == False):
            self.move_towards_goal_alt(x_goal,
                y_goal,
                self.robot_x,
                self.robot_y,
                self.theta,
                self.twist,
                self.pub)
        elif(left == False and right == True):
            self.turn_left(self.twist,self.pub)
        elif(right == False and left == True):
            self.turn_right(self.twist,self.pub)
        else:
            self.calculate_mid(prediction)

    def calculate_mid(self,prediction):
        ''' Calculation whether the first or the third column contain higher
            predictions. prediction[5:32,1:2] means that row 5 to 32 and column
            1 are taken in consideration.
        Args:
            prediction: Models prediction
        '''
        if(prediction[20:41,1:2].mean() < prediction[20:41,3:4].mean()):
            self.turn_left(self.twist,self.pub)
        else:
            self.turn_right(self.twist,self.pub)

# ******************************************************************************
# ************************* Intermediary Goal Algorithm ************************
# ******************************************************************************

    def calc_coordinate_in_robot_system(self,angle_deg,hypotenuse):
        ''' Function calculating x and y value, given the hypotenuse along with 
            an angle. The coordinates are in the robot coordinate system 
            considering the longitudinal axis of the robot as y
        Args:
            hypotenuse: Length from robot of where to calculate coordinate from

        Return:
            Returns the coordinate calculated
        '''
        angle_rad = angle_deg * math.pi / 180
        coordinate = []
        if(angle_rad>0):
            coordinate.append(math.sin(angle_rad)*hypotenuse)
            coordinate.append(math.cos(angle_rad)*hypotenuse)
        else:
            coordinate.append(math.sin(angle_rad)*hypotenuse)
            coordinate.append(math.cos(angle_rad)*hypotenuse)

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

        #Rotation part   
        x_rot = coordinate[0] \
                * math.cos(thetaTemp) \
                - coordinate[1] \
                * math.sin(thetaTemp)
        y_rot = coordinate[0] \
                * math.sin(thetaTemp) \
                + coordinate[1] \
                * math.cos(thetaTemp)
        
        #Translation part
        x_new = robot_x + x_rot
        y_new = robot_y + y_rot

        return x_new, y_new

    def get_goal_coordinates(self,l,m,r,x_main,y_main,pred,n_dist,ratio):
        if(l == False and m == False and r == False):
            return x_main, y_main
        elif(l == False and m == False and r == True):
            coord = self.calc_coordinate_in_robot_system(-25,2.5)
        elif(l == False and m == True and r == True):
            coord = self.calc_coordinate_in_robot_system(-40,2.5)
        elif(l == True and m == False and r == False):
            coord = self.calc_coordinate_in_robot_system(25,2.5)
        elif(l == True and m == True and r == False):
            coord = self.calc_coordinate_in_robot_system(40,2.5)
        elif(l == True and m == True and r == True):
            angle = self.calculate_angle_if_mid_equal(pred,n_dist,ratio)
            coord = self.calc_coordinate_in_robot_system(angle,2.5)
        elif(l == False and m == True and r == False):
            angle = self.calculate_angle_if_mid_equal(pred,n_dist,ratio)
            coord = self.calc_coordinate_in_robot_system(angle,2)
        else:
            x_main, y_main


        return self.get_interm_goal_coordinates(coord,
                                                self.robot_x,
                                                self.robot_y,
                                                self.theta)

    def calculate_angle_if_mid_equal(self,prediction,n_dist,ratio):
        ''' Calculation whether the first or the third column contain higher
            predictions. prediction[5:32,1:2] means that row 5 to 32 and column
            1 are taken in consideration.
        Args:
            prediction: Models prediction
        '''
        upper = int(n_dist * ratio)
        if(prediction[upper:n_dist,1:2].mean() < prediction[upper:n_dist,3:4].mean()):
            return -40
        else:
            return 40

    def get_greenlight(self,prediction,n_dist,ratio,sensitivity):
        ''' Reading prediction and providing greenlight if no obstacles are
            predicted for a specific orientation of the robot. Prediction is
            divided in 5 columns and 31 rows containing predictions about 
            obstacles. The columns represent the 5 laser ranges defined in
            Feature extraction while the rows represent future distance for
            images given.
        Args:
            prediction:     Models prediction
            n_dist:         amount of intermediary distances / positions the model is
                            trained with
            ratio:          ratio to determine the danger zone. If ratio is higher,
                            robot is allowed to approach closer. If it is lower, the
                            evasion techniques are applied earlier.
            sensitivity:    Decides whether the ranges observed should alert or not
        '''
        upper = int(n_dist * ratio)
        if(prediction[upper:n_dist,2:3].mean() < sensitivity):
            mid = False
        else:
            mid = True
        if(prediction[upper+5:n_dist,0:2].mean() < sensitivity):
            left = False
        else:
            left = True
        if(prediction[upper+5:n_dist,3:5].mean() < sensitivity):
            right = False
        else:
            right = True
        rospy.loginfo('%f,%f,%f',left,mid,right)
        return left,mid,right
    
    def get_goal_coordinates_alternative(self,x_main,y_main,n_dist,pred,ratio):
        ''' Function calculating intermediary goal coordinates. At first the
            prediction array is checked for immediate action. Column 2 to 4 are
            observed and their mean calculated. If the mean is smaller than 0.15
            , there is no immediate danger for the robot. If the mean is higher,
            a mean for each entire column is generated and acted upon with
            either sharp or light evasions.
        Args:
            x_main: x coordinate of main goal
            y_main: y coordinate of main goal
            n_dist: amount of intermediary distances / positions the model is
                    trained with
            pred:   models prediction
            ratio:  ratio to determine the danger zone. If ratio is higher,
                    robot is allowed to approach closer. If it is lower, the
                    evasion techniques are applied earlier.
        Return:
            Intermediary goal coordinates
        '''
        upper = int(n_dist * ratio)
        if(pred[upper:n_dist,0:5].mean() < 0.2):
            print("Green")
            return x_main, y_main
        else:
            mean_array = self.get_mean_of_ranges(pred)
            filtered_array = np.where(mean_array < 0.15, False, True)
            print(mean_array)
            print(filtered_array)
            if(np.all(filtered_array)):
                if(mean_array[0] > mean_array[-1]):
                    coord = self.calc_coordinate_in_robot_system(60,2)
                    print("SHARP RIGHT")
                else:
                    coord = self.calc_coordinate_in_robot_system(-60,2)
                    print("SHARP LEFT")
            else:
                if(mean_array[0] > mean_array[-1]):
                    coord = self.calc_coordinate_in_robot_system(30,2.5)
                    print("LIGHT RIGHT")
                else:
                    coord = self.calc_coordinate_in_robot_system(-30,2.5)
                    print("LIGHT LEFT")
            
            return self.get_interm_goal_coordinates(coord,
                                                    self.robot_x,
                                                    self.robot_y,
                                                    self.theta)

    def get_min_column(self,pred):
        ''' Function returning the column with the minimum mean of n ranges.
            The function is currently not used as argmin can be ambiguous.
        Args:
            pred: models prediction
        Return:
            Index of column with min mean of ranges
        '''
        temp = np.empty([1,5])
        for i in range(len(pred[0])):
            temp[0][i] = pred[:,i].mean()
        print(temp)
        return np.argmin(temp)

    def get_mean_of_ranges(self,pred):
        ''' Function calculates an array with the mean for each range
        Args:
            pred: models prediction
        Return:
            Array with mean for each range / column
        '''
        temp = np.empty([5,])
        for i in range(len(pred[0])):
            temp[i] = pred[:,i].mean()
        return temp



# ******************************************************************************
# ************************* Main Loop ******************************************
# ******************************************************************************

    def main(self):
        ''' Function containing the main loop for this node. Beside a heatmap
            activated displaying predictions made by the model, the loop 
            iterates over coordinates while trying to avoid obstacles.
            Distances about obstacles in front of the robot, are provided by
            a trained model given merely an image. A laser is just used for
            debugging and testing purposes. The environment for this script can
            be loaded with: 
            rosrun gazebo_ros spawn_model -file /root/catkin_ws/src/
            ss20_lanz_2d_obstacle_avoidance/source/files/environments/final/
            model.sdf -sdf -model model7
        '''
        #Total amount of rows. Needs to be changed for different models
        n_dist = 35

        #Determines how many rows are used
        ratio = 0.75

        #Sensitivity to decide whether a calculation should be triggered or not
        sensitivity = 0.35

        state = 3
        counter = 0
        substate = 0
        

        FA = FileAdministration(ros = True)  
             
        idx = FA.select_model()
        print('Model index !')
        print(idx)
        model_dir = FA.get_pipeline_directory(3,idx)

        t = Train()
        cnn = t.model(n_dist, old_version=False)
        cnn.load_weights(model_dir + 'model_' + str(idx) + '.h5')

        ''' Set numpy print options
        '''
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        fig, axim = self.numpy_to_heatmap_setup(n_dist)

        ranges = self.get_ranges(200,5,666)

        list_coordinates = np.array([[10, 0], [10, 10],[0,10],[0,0],[-10,0],
                                     [-10,10],[0,10],[0,0],[-10,0],[-10,-10],
                                     [0,-10],[0,0],[10,0],[10,-10],[0,-10],
                                     [0,0]])

        while not rospy.is_shutdown():
            ''' Added one axis as cnn.predict(param) expects a 4 dimensional 
                array
            '''
            data_compressed_plus_axis = self.jpeg_to_numpy(self.data_compressed,
                                                           (80,64), 
                                                           True)[np.newaxis]
            
            ''' Predicts laser values giving as input an image. Returns array of 
                shape (1,155)
            '''
            prediction = cnn.predict(data_compressed_plus_axis)

            ''' Reshaping the array. -1 stands for a value which is 
                automatically calculated. Reshaped array: (1, n_dist, 5)
            '''
            pred = prediction.reshape([1,-1,5])[:,:n_dist,:]

            ''' Reversing the order of the columns and rows
            '''
            pred = np.squeeze(pred[:,::-1,::-1], axis=0)

            ''' Create heatmap. To get matplotlib working inside docker you need
                to implement following line in the docker file:
                RUN sudo apt-get install tcl-dev tk-dev python-tk python3-tk
            '''
            
            axim.set_data(pred*100)
            fig.canvas.flush_events()

            ''' Function used for testing and debugging purposes

            '''
            self.calculate_ground_truth(ranges)

            if(state == 0):
                x_main, y_main = self.get_current_coordinate(counter,
                                                             list_coordinates)
                state = 1
            if(state == 1):
                rospy.loginfo('Robot at state %f',state)

                x,y = self.get_goal_coordinates_alternative(x_main,y_main,n_dist,pred,ratio)
                
                state = 2
            if(state == 2):
                rospy.loginfo('Robot at state %f',state)
                if(self.distance(self.robot_x,self.robot_y,x,
                                y) < 0.2):
                    self.halt_robot()
                    if(x != x_main and y != y_main):
                        state = 1
                    else:
                        state = 0
                        if(counter == 15):
                            counter = 0
                        else:
                            counter = counter + 1
                else:
                    if(x != x_main and y != y_main):
                        state = 2
                    else:
                        state = 1
                    self.move_towards_goal_alt(x,y,self.robot_x,self.robot_y,
                                                self.theta,
                                                self.twist,
                                                self.pub)
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('deploying_model')
    Deployment()
    rospy.spin()