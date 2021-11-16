import cv2
import numpy as np
import os
import argparse
import dtcwt

class LLenhancement:
    def __init__(self,args):
        self.orig_img = cv2.imread(args.img_path)
        if self.orig_img is not None:
            self.orig_hsv = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2HSV)
            self.v = np.array(self.orig_hsv[:,:,2])
        else:
            raise ValueError("Invalid image path")

    def lowpassEnhancement(self,lowpass_img):
        #Global adaptation


        #Local adaptation

        return 

    def highpassEnhancement(self,highpass_imgs):
        #iterate through 6 sub bands
        return

    def whiteBalance(self,v_channel):
        return

    def imgEnhancement(self):
        transform = dtcwt.Transform2d()
        fwd_tfm = transform.forward(self.v, nlevels=1)
        # (N,N)
        fwd_tfm.lowpass = self.lowpassEnhancement(fwd_tfm.lowpass)
        # (N/2,N/2,6)
        fwd_tfm.highpasses[0] = self.highpassEnhancement(fwd_tfm.highpasses[0])
        inv_tfm = transform.inverse(fwd_tfm)
        white_balanced = self.whiteBalance(inv_tfm)

        self.enhanced_hsv = np.array(self.orig_hsv)
        self.enhanced_hsv[:,:,2] = white_balanced
        self.enhanced_bgr = cv2.cvtColor(self.enhanced_hsv, cv2.COLOR_HSV2BGR)

        return self.enhanced_bgr


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Low Light Image Enchancement')
    parser.add_argument('--img_path',type = str,default='./img1.png',help='Path to low light image to be enhanced')
    parser.add_argument('--save_img',type = str,default='./enhanced_img1.png',help='Save enhanced image as')
    args = parser.parse_args()
    enhancer = LLenhancement(args)
    enhanced_img = enhancer.imgEnhancement()
    cv2.imwrite(args.save_img, enhanced_img)


# img = cv2.imread('./img1.png')
# imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# v = np.array(imghsv[:,:,2])
# transform = dtcwt.Transform2d()
# fwd_tfm = transform.forward(v, nlevels=2)
# print(v.shape)
# print(fwd_tfm.lowpass.shape)
# # print(fwd_tfm.l)
# print(fwd_tfm.highpasses[1].shape)
