import os
import cv2
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
            return 'CustomError, {0}{1} !'.format(self.message,self.idx)
        else:
            return 'CustomError has been raised'

class VisualizeBagWithPrediction():
    ''' Visualization a HDF5 instance along with the prediction made by the 
        model.
    '''
    def draw_image(self, img_frame, offset, spacing, height, width, readings, 
                    is_pred):
        ''' Draws the prediction of the model and its corresponding ground truth
            as rectangles

        Args:
            img_frame: 	numpy array containing the raw image
            offset: 	a tuple containing vertical and horizontal offset
            spacing: 	a number representing the spacing between each sensor 
                        rectangle
            height: 	the height of the rectangle
            width: 		the width of the rectangle
            readings: 	a list of readings
            is_pred: 	boolean flag representing if readings are taken from the 
                        model or from the sensors.
        '''
        if is_pred:
            a = np.array([230, 230, 255])
            b = np.array([25, 25, 150])
        else:
            readings = (readings > 0).astype(np.float)
            a = np.array([255, 230, 230])
            b = np.array([150, 50, 50])

        for i in range(len(readings)):
            cv2.rectangle(img_frame,
                            (offset[0] \
                            + spacing + (spacing + width) * i, offset[1]),
                            (offset[0] + width + spacing + (spacing + width) \
                            * i, offset[1] + height),
                            self.linear_interpolation(a, 
                                                      b, 
                                                      readings[i]).tolist(), 
                                                      cv2.FILLED)

    def linear_interpolation(self, x, y, a):
        '''Linear interpolation between two points by a fixed amount.

        Args:
            x: first point.
            y: second point.
            a: percentage between the two points.

        Returns:
            the interpolated point.
        '''
        return (1 - a) * x + a * y

    def visualize_output(self):
        ''' Visualize the content of a specific bag within the HDF5 file with
            its corresponding prediction by the model. After launching the file,
            the user can chose which model to create the visualization from.
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--mode',type=int,
                            help='autonomous [0] or normal mode [1]',
                            default=1)
        parser.add_argument('-bi', '--bagindex',type=int,
                            help='specifiy bag index to create file from',
                            default=1)
        parser.add_argument('-tc','--target-count',type=int,
                            help='amount of positions in dm of which to relate',
                            default=35)

        args = parser.parse_args()
        mode = args.mode
        bag_index = args.bagindex
        target_count = args.target_count

        t = Train()

        FA = FileAdministration()
        idx = FA.get_pipeline_index(current=True)
        model_dir = FA.get_pipeline_directory(3,idx)
        dataset_dir = FA.get_pipeline_directory(2,idx)
        model_video_dir = FA.get_pipeline_directory(5,idx)

        ''' Returns images as x and their corresponding ground truth as y
            x has the shape (60, 64, 80, 3) and y has (60, 155)
        '''
        x, y, _ = next(t.generator(dataset_dir +'dataset_'+str(idx)+'.h5',
                      [bag_index], 
                      32, 
                      is_testset=True, 
                      augment=False, 
                      do_flip=False))
        l = x.shape[0]

        #Reshaping of y to (60, 31, 5)
        y = y.reshape([l, -1, 5])[:, :31, :]
        #31
        d = y.shape[1]

        print('Generating predictions...')
        
        cnn = t.model(target_count, old_version=False)
        cnn.load_weights(model_dir + 'model_' + str(idx) + '.h5')

        #Loading the predicitons of shape (60, 155)
        pred = cnn.predict(x)

        spacing = 5
        height = int(np.floor(200.0 / d))
        width = 30
        video = cv2.VideoWriter(model_video_dir 
                                + 'model_video[' 
                                + str(bag_index) 
                                + ']_' + str(idx) 
                                + '.avi', 
                                cv2.VideoWriter_fourcc(*'XVID'), 
                                10, 
                                (400, 300 + 220))

        print('Video creation...')

        ''' Here the images are drawn. The outer loop iterates 60 and the inner
            loop 31 times. For every row, each iterating 5 elements are selected 
            consecutevily. Each batch stands for the distance containing the 
            values of each range. The outer loop is iterating over each row.
        '''
        for i in range(l):
            img_frame = cv2.resize(x[i], (400, 300))
            img_frame = (img_frame - img_frame.min()) / (img_frame.max() - img_frame.min())
            img_frame = (img_frame * 255).astype(np.uint8)
            img_frame = np.vstack([np.ones((220, 400, 3), np.uint8) * 255, img_frame])

            for j in range(d):
                #Prediction drawn
                self.draw_image(img_frame, (0, 200 - j * height), 
                                spacing, 
                                height, 
                                width,
                                pred[i, j * 5:(j + 1) * 5],
                                True)
                #Ground truth drawn
                if y[i, j, 0] != -1:
                    self.draw_image(img_frame, 
                                    (220, 200 - j * height), 
                                    spacing, 
                                    height, 
                                    width, 
                                    y[i, j, :], 
                                    False)
                else:
                    pass
                    cv2.rectangle(img_frame,
                        (225, 205 - j * height),
                        (395, 205 - (j + 1) * height),
                        (128, 128, 128), cv2.FILLED)

            video.write(img_frame)

            #cv2.imshow('img_frame', img_frame)
            if cv2.waitKey(1000 // 10) & 0xFF == ord('q'):
                break

        video.release()

        cv2.destroyAllWindows()

if __name__ == '__main__':
	v = VisualizeBagWithPrediction()
	v.visualize_output()
