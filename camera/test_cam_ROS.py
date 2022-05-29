import cv2
from sensor_msgs.msg import Image
import rospy
import numpy as np

class CameraTest(object):
    def __init__(self):
        self.camera_topic = '/usb_cam/image_raw'
        self.image_sub = rospy.Subscriber( self.camera_topic, Image, self.image_callback)
        self.image = None
        

    def image_callback(self,msg):
        self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        #print(self.image.shape)
        cv2.imshow('asdf',self.image)
        cv2.waitKey(1)

        
            
if __name__ == '__main__':
    try:
        rospy.init_node('cam_reader', anonymous=True)
        
        lane = CameraTest()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
   
