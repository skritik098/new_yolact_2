import datetime
import time
import numpy as np
import cv2
from collections import deque
from scipy.ndimage.measurements import label
from PIL import Image
from keras import backend as K
from utils import img_utils, lane_utils, car_utils, yolo_utils
from utils.lane_utils import Line

##Create an image processor class
class ImageProcessor: 
    
    def __init__(self, img_points, obj_points, mtx, dist, src, dst):
        
        #General Attributes
        self.frame = 0
        #self.log = open(".\\log\\detector_log.txt","w+")
        self.undistort_time = 0
        
        #Attributes for Lane detection

        self.left_line = Line()
        self.right_line = Line()
        self.img_points = img_points
        self.obj_points = obj_points
        self.mtx = mtx
        self.dist = dist
        self.src = src
        self.dst = dst
        self.offset = deque(maxlen=self.left_line.smooth_count)
        self.L_thresh = (215,255)
        self.B_thresh = (150,255)
        self.lane_time = 0
        self.binary_time = 0
        self.draw_time = 0
        self.lane_total = 0
        
    def lane_detection(self, img):
        self.frame += 1
        z_score1 = 1.9
        tb1 = time.time()
        
        #Create binary
        binary = lane_utils.create_binary(img, self.L_thresh, self.B_thresh)
    
        #Warp Image
        warped_binary = img_utils.warp_image(binary, self.src, self.dst)
        tb2 = time.time()
        self.binary_time += (tb2-tb1)
        tl1 = time.time()
        #Find centroids
        self.left_line.centers, self.right_line.centers = lane_utils.find_centers(warped_binary)
        
        #self.log.write('{} ############# NEXT FRAME ##############\n'.format(datetime.datetime.now()))
        #self.log.write('{} FRAME {:d} - # of LEFT centers is: {:d}\n'.format(datetime.datetime.now(), 
        #                                                                     self.frame, self.left_line.centers.shape[0]))
        #self.log.write('{} FRAME {:d} - # of RIGHT centers is: {:d}\n'.format(datetime.datetime.now(), 
        #                                                                      self.frame, self.right_line.centers.shape[0]))
        #Get line co-efficients if more than three centers are found for each line:
        if (self.left_line.centers.shape[0] > 2 ) & (self.right_line.centers.shape[0] > 2):
            
            #self.log.write('{} ENTERED IF BLOCK FOR FRAME: {}\n'.format(datetime.datetime.now(), self.frame))
            #self.log.write('{} Left centers shape: {}\n'.format(datetime.datetime.now(), self.left_line.centers.shape[0]))
            #self.log.write('{} Right centers shape: {}\n'.format(datetime.datetime.now(), self.right_line.centers.shape[0]))
            #self.log.write('{} Left centers are: {}\n'.format(datetime.datetime.now(), self.left_line.centers))
            #self.log.write('{} Right centers are: {}\n'.format(datetime.datetime.now(), self.right_line.centers))
            
            self.left_line.centers, self.right_line.centers = lane_utils.check_centers(self.left_line.centers, 
                                                                                       self.right_line.centers, z_score1)
            #self.log.write('{} CHECK CENTERS CALLED\n'.format(datetime.datetime.now()))
            #self.log.write('{} Left centers shape: {}\n'.format(datetime.datetime.now(), self.left_line.centers.shape[0]))
            #self.log.write('{} Right centers shape: {}\n'.format(datetime.datetime.now(), self.right_line.centers.shape[0]))
        
            #self.log.write('{} Getting Left Line Co-efficients\n'.format(datetime.datetime.now()))
            self.left_line.coeff = lane_utils.get_coeff(self.left_line.centers)
            #self.log.write('{} Getting Right Line Co-efficients\n'.format(datetime.datetime.now()))
            self.right_line.coeff = lane_utils.get_coeff(self.right_line.centers)
            #self.log.write('{} GOT CO-EFFICIENTS\n'.format(datetime.datetime.now()))
            self.left_line.coeff_list.append(self.left_line.coeff)
            self.right_line.coeff_list.append(self.right_line.coeff)
            #self.log.write('{} CO-EFFICIENTS APPENDED\n'.format(datetime.datetime.now()))

            #Measure ROC and Offset
            l_ROC, r_ROC, current_offset = lane_utils.get_ROC_offset(img, self.left_line.centers, 
                                                                                        self.right_line.centers)
            self.left_line.ROC.append(l_ROC)
            self.right_line.ROC.append(r_ROC)
            self.offset.append(current_offset)
            
        avg_ROC = (np.mean(self.left_line.ROC) + np.mean(self.right_line.ROC))/2
        avg_offset = np.mean(self.offset)
        tl2 = time.time()
        self.lane_time += (tl2 - tl1)
        #TO-DO: Sanity checks
        #Check that ROCs for left_line & right_line in this frame are within a threshold of ROC of average ROC
        #Check that left_line and right_line are equidistant in warped_binary at bottom, middle & top of frame
        #If sanity checks fail, remove the latest appended offset, ROC and co-efficients
        
        td1 = time.time()
        #Draw Lines & measure RoC
        warped_img = img_utils.warp_image(img, self.src, self.dst)
        draw_img = lane_utils.draw_lines(warped_img, self.left_line.best_fit(), self.right_line.best_fit())
        
        #Unwarp Image & add to original
        ##diag_img_size = 250
        ##stacked_warped_binary = np.dstack((warped_binary*255, warped_binary*255, warped_binary*255)).astype(np.uint8)
        unwarped_draw_img = img_utils.warp_image(draw_img, self.dst, self.src)
        output_img = cv2.addWeighted(img, 1, unwarped_draw_img, 0.5, 0)
        ##diag_img = cv2.resize(cv2.addWeighted(stacked_warped_binary, 1, draw_img, 0.9, 0), 
        ##                      (diag_img_size, diag_img_size), interpolation=cv2.INTER_AREA) 
        ##output_img[:diag_img_size,(1280-diag_img_size):1280,:] = diag_img
        
        #Add text to image
        cv2.putText(output_img,"Frame: {:d}".format(self.frame), (50,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(output_img,"Offset: {:0.2f} m.".format(avg_offset), (50,110), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255,0,0), 2, cv2.LINE_AA)
        if avg_ROC>4000:
            cv2.putText(output_img,"RoC: Straight", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255,0,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(output_img,"RoC: {:0.2f} m.".format(avg_ROC), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255,0,0), 2, cv2.LINE_AA)
        
        #self.log.write('{} COMPLETED PROCESSING FRAME\n'.format(datetime.datetime.now()))
        td2 = time.time()
        self.draw_time += (td2 - td1)
        
        return output_img