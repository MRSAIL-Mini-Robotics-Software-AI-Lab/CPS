"""
Encode and decode classes for every pixel in the RGB channels
"""
import numpy as np
import cv2
import typing

class CPS:
    def __init__(self,noOfBits=2) -> None:
        self.nOfBits = noOfBits
        self.classBits = self.nOfBits * 3
        
    def encode(self,img:np.ndarray,classes:np.ndarray)->np.ndarray:
        """
        Encode classes into RGB channels
        Parameters
        ----------
        img (3,H,W): np.ndarray
            RGB image 
        classes (H,W): np.ndarray
            Classes for every pixel
        
        Returns
        -------
        img (3,H,W): np.ndarray
            RGB image with classes encoded
        """
        assert img.shape[1:] == classes.shape, "Image and classes must have the same shape"
        assert len(img.shape) == 3 , "Image must have 3 channels"
        assert len(classes.shape) == 2 , "Classes must have 2 channels"
        assert img.shape[0] == 3, "Image must have 3 RGB channels in shape (3,H,W)"
        assert classes.dtype == np.uint8 , "Classes must be of type np.uint8"
        assert img.dtype == np.uint8 , "Image must be of type np.uint8"

        assert np.max(classes) < 2**self.classBits, "Classes must be less than 2**classBits"
        assert np.min(classes) >= 0, "Classes must be greater than 0"
        print("Classes : ",classes)
        print("Image : ",img)

        lastSectionBits = classes%(2**self.nOfBits)
        middleSectionBits = (classes//(2**self.nOfBits))%(2**self.nOfBits)
        firstSectionBits = (classes//(2**(self.nOfBits*2)))%(2**self.nOfBits)

        newR = img[0,:,:]
        newR = newR//(2**self.nOfBits)
        newR = newR*(2**self.nOfBits)
        newR = newR + lastSectionBits

        newG = img[1,:,:]
        newG = newG//(2**self.nOfBits)
        newG = newG*(2**self.nOfBits)
        newG = newG + middleSectionBits

        newB = img[2,:,:]
        newB = newB//(2**self.nOfBits)
        newB = newB*(2**self.nOfBits)
        newB = newB + firstSectionBits

        img[0,:,:] = newR
        img[1,:,:] = newG
        img[2,:,:] = newB
        print("New Image : ",img)
        return img
    
    def decode(self,points):
        """
        Decode classes from colored point cloud
        Parameters
        ----------
        points (N,6): np.ndarray
            Colored point cloud (x,y,z,r,g,b)

        Returns
        -------
        classes (N,1): np.ndarray
            Classes for every pixel
        """
        assert points.shape[1] == 6, "Points must have 6 channels"
        assert points.dtype == np.uint8, "Points must be of type np.uint8"
        print("Points : ",points)
        lastSectionBits = points[:,3]%(2**self.nOfBits)
        middleSectionBits = (points[:,4])%(2**self.nOfBits)
        firstSectionBits = (points[:,5])%(2**self.nOfBits)

        classes = firstSectionBits*(2**(self.nOfBits*2)) + middleSectionBits*(2**self.nOfBits) + lastSectionBits
        print("Classes : ",classes)
        return classes

if __name__ == "__main__":
    cps = CPS(2)
    img = np.random.randint(0,244,(3,10,10),dtype=np.uint8)
    classes = np.random.randint(0,64,(10,10),dtype=np.uint8)
    points = np.random.randint(0,244,(100,6),dtype=np.uint8)
    
    encoded = cps.encode(img,classes)
    points[:,3] = encoded[0,:,:].flatten()
    points[:,4] = encoded[1,:,:].flatten()
    points[:,5] = encoded[2,:,:].flatten()
    decoded = cps.decode(points)
    print("Classes : ",classes)
    print("Decoded : ",decoded)