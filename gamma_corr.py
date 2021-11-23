import cv2
import numpy as np
import os
import argparse
import sys

def gamma_correct(img_path, gamma=0.6):
    """
    inputs - 
        img_path - path to source image
        gamma - factor for gamma transform
    outputs - 
        img - gamma_corrected image
    """
    img = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    values = img_hsv[:,:,2]
        
    K=np.power(255, gamma)
    # pixel_i = pixel_i^gamma, normalized by the value of the highest pixel
    values = np.power(values, gamma)/K
    # normalize for display
    values = values*255
    img_hsv[:,:,2]=values
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return img

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Vanilla Gamma Transform')
    parser.add_argument('--img_path',type = str,default='./img1.png',help='Path to low light image to be enhanced')
    parser.add_argument('--save_img',type = str,default='./enhanced_img1.png',help='Save enhanced image as')
    parser.add_argument('--gamma',type = float,default= 0.6,help='Gamma to use for gamma correction')
    args = parser.parse_args()
    enhanced_img = gamma_correct(args.img_path, args.gamma)
    cv2.imwrite(args.save_img, enhanced_img)