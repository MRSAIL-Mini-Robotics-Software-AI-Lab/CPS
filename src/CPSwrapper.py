#!/usr/bin/env python3

"""
ROS wrapper for CPS
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from CPS import CPS
from cv_bridge import CvBridge 

class CPSWrapper:
    
    def __init__(self,noOfBits,image_topic,encoded_images,classes_topic=None) -> None:
        self.nOfBits = noOfBits
        self.classBits = self.nOfBits * 3
        self.cps = CPS(noOfBits=self.nOfBits)
        self.image_topic = image_topic
        self.encoded_images = encoded_images
        self.classes_topic = classes_topic
        self.cv_bridge = CvBridge()
    def create_publisher(self):
        self.image_pub = rospy.Publisher(self.encoded_images,Image,queue_size=10)
        if self.classes_topic is not None:
            self.classes_pub = rospy.Publisher(self.classes_topic,Image,queue_size=10)

    def imageCallback(self,image):
        #convert Image to numpy
        image = np.frombuffer(image.data,dtype=np.uint8).reshape(image.height,image.width,3)
        image = image.copy()
        image = np.transpose(image,(2,0,1))
        #Random classes
        classes = np.random.randint(0,2**self.classBits,size=(image.shape[1],image.shape[2]),dtype=np.uint8)
        #Encode
        encoded_image = self.cps.encode(image,classes)
        #Publish
        encoded_image = np.transpose(encoded_image,(1,2,0))
        encoded_msg = self.cv_bridge.cv2_to_imgmsg(encoded_image)
        self.image_pub.publish(encoded_msg)

        #Verify the classes
        targetClasses = classes.flatten()

        #Simulate point cloud
        points = np.zeros((image.shape[1]*image.shape[2],6),dtype=np.uint8)
        points[:,3] = encoded_image[:,:,0].flatten()
        points[:,4] = encoded_image[:,:,1].flatten()
        points[:,5] = encoded_image[:,:,2].flatten() 
        decodedClasses = self.cps.decode(points)

        #Verify
        assert np.all(targetClasses == decodedClasses), "Classes do not match"






    def startModule(self):
        self.create_publisher()
        rospy.init_node("CPSWrapper",anonymous=True)
        rospy.Subscriber(self.image_topic,Image,self.imageCallback)
        rospy.spin()

if __name__ == "__main__":
    wrapper = CPSWrapper(2,image_topic="/front_camera_rgb/image_raw",encoded_images="/cps/image")
    wrapper.startModule()