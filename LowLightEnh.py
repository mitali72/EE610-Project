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
        '''
        input 
            low-pass sub-image from dt-cwt
        output
            adaptive local tone mapping applied on low-pass sub-image
        '''
        Lw = np.array(lowpass_img)
        h,w = Lw.shape[0],Lw.shape[1]
        #Global adaptation
        logmeanL = np.exp(np.sum(np.log(1e-5 + Lw))/(h*w))
        Lwmax = np.max(Lw)
        Lg = np.log((Lw/logmeanL)+1)/(np.log((Lwmax/logmeanL)+1))

        #Local adaptation
        Hg = np.zeros((h,w))
        kernel_size = 3
        Lgpad = np.pad(Lg,(kernel_size-1)//2,'constant')
        a = np.zeros((h,w))
        b = np.zeros((h,w))
        
        #Guided filter, guidance image = input image
        for i in range(Lgpad.shape[0]-kernel_size):
            for j in range(Lgpad.shape[1]-kernel_size):

                Lwindow = Lgpad[i:i+kernel_size][j:j+kernel_size]
                sigma = np.mean(np.square(Lwindow)) - np.sqaure(np.mean(Lwindow))
                a[i][j] = sigma/(sigma + 0.01)
                b[i][j] = (1-a[i][j])*np.mean(Lwindow)

        for i in range(w):
            for j in range(h):
                Hg[i][j] = np.mean(a[max(i-kernel_size//2,0):min(i+1+kernel_size//2,w),max(j-kernel_size//2,0):min(j+1+kernel_size//2,h)])

        alpha = 1+36*(Lg/np.max(Lg))
        beta = 10*np.exp(np.sum(np.log(1e-5 + Lg))/(h*w))
        Lout = alpha*np.log((Lg/Hg)+beta)
        return Lout

    def highpassEnhancement(self,highpass_imgs, T=1e-6):
        """
        input
            highpass_imgs - highpass coeffs from dt-cwt
            T - threshold value for soft thresholding
        output - highpass coeffs after soft thresholding
        """

        for idx in range(highpass_imgs.shape[2]):
            a = np.absolute(highpass_imgs[:,:,idx]) - T
            highpass_imgs[:,:,idx] = np.sign(highpass_imgs[:,:,idx])*a*(a>0)

            highpass_imgs[:,:,idx] += highpass_imgs[:,:,idx]*(a<0)
        return highpass_imgs

    def whiteBalance(self, v_channel, s1=0.1, s2=0.1):
        """
        inputs
            v_channel - v channel of image (assuming values between 0 and 255)
            s1 - low threshold percentage
            s2 - high threshold percentage
        output - white-balanced v channel
        """
        # v_channel = v_channel.astype(int)
        v_channel = np.clip(v_channel, 0, 255)
        histo = np.zeros(256)
        rowsize = v_channel.shape[0]
        colsize = v_channel.shape[1]
        
        # compute histogram, add one at each place whenever corresponding value appears
        for i in range(rowsize):
            for j in range(colsize):
                histo[int(v_channel[i,j])]+=1
        # histogram normalized by number of pixels
        histo/=(rowsize*colsize)

        # cdf(i) = histo(i)+histo(i-1)+....+histo(1)
        cdf = np.array([np.sum(histo[:i+1]) for i in range(256)])

        # compute s1th and 1-s2th percentile values
        vmin = 0
        while(cdf[vmin + 1] <= s1):
            vmin+=1
        vmax = 255-1
        while(cdf[vmax - 1] >= 1-s2):
            vmax-=1
        if (vmax < (255 - 1)):
            vmax+=1

        # saturate the pixels outside the given range
        v_channel = (v_channel<=vmin)*vmin + v_channel*(v_channel>vmin)
        v_channel = (v_channel>=vmax)*vmax + v_channel*(v_channel<vmax)

        # rescale 
        v_channel = 255.0*(v_channel-vmin)/(vmax-vmin)

        return v_channel

    def imgEnhancement(self):
        transform = dtcwt.Transform2d()
        fwd_tfm = transform.forward(self.v, nlevels=1)
        # (N,N)
        fwd_tfm.lowpass = self.lowpassEnhancement(fwd_tfm.lowpass)
        # (N/2,N/2,6)
        fwd_tfm.highpasses = (self.highpassEnhancement(fwd_tfm.highpasses[0]),)
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
