import rospy
from sensor_msgs.msg import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Camera:
    def __init__(self):
      
        self.image_sub_R = rospy.Subscriber('/mynteye/right/image_raw', Image, self.image_callback_R)
        self.image_sub_L = rospy.Subscriber('/mynteye/left/image_raw', Image, self.image_callback_L)
        self.image_L = None
        self.image_R = None
   

    def image_callback_R(self,msg):
        
        self.image_R = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        

    def image_callback_L(self,msg):
        
        self.image_L= np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        
    def display_image(self):
          
        if self.image_L is not None:
#             plt.subplot(1,2,1)
#             plt.cla()
#             plt.imshow(self.image_L,cmap = 'gray')
            cv2.imshow('left', self.image_L)
#             plt.subplot(1,2,2)
#             plt.cla()
#             plt.imshow(self.image_R,cmap = 'gray')
            cv2.imshow('Right', self.image_R)
            cv2.waitKey(1)
    
        
#             plt.pause(0.01)

if __name__ == '__main__':
    try:
        rospy.init_node('stereo_cam')       

        cam = Camera()
        while not rospy.is_shutdown():
            cam.display_image()

    except rospy.ROSInterruptException:
        pass







