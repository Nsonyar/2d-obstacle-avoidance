import os
import cv2
import h5py
import argparse
import numpy as np
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

class VisualizeDataset():
    ''' Creates and visualizes the HDF5 file content as a video.
    '''

    def draw_image(self, img_frame, prox, bar_w, bar_h, off_x, size):
        '''Draws the sensor readings onto the image using rectangles

        Args:
            img_frame: a numpy array containing the raw image
            prox: a list of readings
            bar_w: the width of the rectangle
            bar_h: the height of the rectangle
            off_x: a number representing the horizontal offset
            size: the size of the img_frame
        '''
        maxx = 4700
        cv2.rectangle(img_frame,
                    (off_x, size[1] - int(5 + bar_h * prox[0] / maxx)),
                    (off_x + bar_w, size[1]),
                    (0, int(255 - (prox[0] / maxx) * 255), 255), cv2.FILLED)
        cv2.rectangle(img_frame,
                    (off_x + bar_w, size[1] - int(5 + bar_h * prox[1] / maxx)),
                    (off_x + bar_w * 2, size[1]),
                    (0, int(255 - (prox[1] / maxx) * 255), 255), cv2.FILLED)
        cv2.rectangle(img_frame,
                    (off_x + bar_w * 2, size[1] - int(5 + bar_h * prox[2] / maxx)),
                    (off_x + bar_w * 3, size[1]),
                    (0, int(255 - (prox[2] / maxx) * 255), 255), cv2.FILLED)
        cv2.rectangle(img_frame,
                    (off_x + bar_w * 3, size[1] - int(5 + bar_h * prox[3] / maxx)),
                    (off_x + bar_w * 4, size[1]),
                    (0, int(255 - (prox[3] / maxx) * 255), 255), cv2.FILLED)
        cv2.rectangle(img_frame,
                    (off_x + bar_w * 4, size[1] - int(5 + bar_h * prox[4] / maxx)),
                    (off_x + bar_w * 5, size[1]),
                    (0, int(255 - (prox[4] / maxx) * 255), 255), cv2.FILLED)

    def visualize(self):
        ''' Creates and visualizes the HDF5 file content.
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--mode',type=int,
                            help='autonomous [0] or normal mode [1]',
                            default=1)
        args = parser.parse_args()
        mode = args.mode

        FA = FileAdministration()
        idx = FA.get_pipeline_index(current=True)
        dataset_dir = FA.get_pipeline_directory(2,idx)
        dataset_video_dir = FA.get_pipeline_directory(6,idx)


        h5f = h5py.File(dataset_dir +'dataset_'+str(idx)+'.h5', 'r')
        bags =  np.unique([str(b[:-2]) for b in h5f.keys()])

        for bag in bags:
            print('Found ' + bag)
            
            Xs = h5f[bag + '_x'][:]
            Ys = h5f[bag + '_y'][:]
            l = Xs.shape[0]

            size = (400, 300)
            bar_h = 80
            bar_w = 80
            off_x = 0
            video = cv2.VideoWriter(dataset_video_dir \
                                    + 'dataset_video[' \
                                    + bag \
                                    + ']_'+ str(idx) +'.avi', 
                                    cv2.VideoWriter_fourcc(*'XVID'), 6, size)

            for i in range(l):
                x = Xs[i]
                y = Ys[i]
                
                img_frame = cv2.resize(x, size)
                img_frame = (img_frame - img_frame.min()) / (img_frame.max() - img_frame.min())
                img_frame = (img_frame * 255).astype(np.uint8)

                self.draw_image(img_frame, y[0:5], bar_w, bar_h, off_x, size)

                video.write(img_frame)

                #cv2.imshow(bag, img_frame)
                if cv2.waitKey(1000 // 10) & 0xFF == ord('q'):
                    break

        video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    v = VisualizeDataset()
    v.visualize()
