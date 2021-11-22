from typing import DefaultDict
import cv2
import numpy as np
import argparse
from helper import *

class LLenhance():

    def __init__(self,args,alpha,gamma):
        self.orig_img = cv2.imread(args.img_path).astype(np.float32)
        if self.orig_img is None:
            raise ValueError("Invalid image path")
            # self.orig_hsv = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2HSV)
            # self.v = np.array(self.orig_hsv[:,:,2])

    def fwdDiffToep(m):
        D = np.zeros((m,m))
        for i in range(m-1):
            D[i,i] = -1
            D[i,i+1] = 1

        D[m-1][0] = 1
        D[m-1][m-1] = -1
        return D

    def gradT(T):
        return cv2.Sobel(T,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(T,cv2.CV_32F,0,1,ksize=3)

    def Tdenom(mu,m,n):
        Dh = np.zeros((m,n))
        Dv = np.zeros((m,n))
        Dh[1,1] = -1
        Dh[1,2] = 1
        Dv[1,1] = -1
        Dv[2,1] = 1
        Dhfft = np.fft.fft2(Dh)
        Dvfft = np.fft.fft2(Dh)
        denom = 2 + mu*(Dhfft*np.conjugate(Dhfft) + Dvfft*np.conjugate(Dvfft))
        return denom
    
    def refineT(self,Tinit,alpha = 0.15,rho = 1.1):

        m,n = Tinit.shape[0],Tinit[1]
        #Initialising G: gradT, Z: Lagrangian multiplier
        G = np.zeros((2*m,n))
        Z = np.zeros((2*m,n))
        #Initialising weight matrix using strategy II
        delTih,delTiv = self.gradT(Tinit)
        Wh = 1/(np.abs(delTih)+1e-5)
        Wv = 1/(np.abs(delTiv)+1e-5)
        W = np.concatenate(Wh,Wv)

        mu = 0.05
        delta = 1e-5
        max_iter = 100
        #discrete gradient operators with forward difference
        Dv = self.fwdDiffToep(m)
        Dh = self.fwdDiffToep(n).T

        while(max_iter):
            #T sub-problem
            x = G - Z/mu
            num = np.fft.fft2(2*Tinit + mu*(np.matmul(x[:m,:],Dh)+np.matmul(Dv,x[m:,:])))
            denom = self.Tdenom(mu,m,n)
            T = np.fft.ifft2(num/denom)

            #G sub-problem
            delTh,delTv = self.gradT(T)
            delT = np.concatenate(delTh,delTv)
            epsilon = alpha*W/mu
            x = delT +Z/mu
            G = np.sign(x)*np.clip(np.abs(x)- epsilon,a_min = 0,a_max = None)

            #Updating Z and mu
            Z = Z + mu*(delT - G)
            mu = mu*rho

            #check convergence
            if(np.linalg.norm(delT-G,'fro')< delta*np.linalg(Tinit,'fro')):
                break

        return T

    def enhanceLL(self,alpha,gamma):
        '''
        input: 
            alpha
            gamma
        output:
            Enhanced image
        '''
        #Estimate initial illumination map
        Tinit = initial_map(self.orig_img)
        Tinit /= 255
        #Refine illumination map
        T = self.refineT(Tinit,alpha,rho = 1.1)
        #Gamma correction
        Tg = gamma_correct(T,gamma)
        Tg = Tg*255
        #Denoising and Recomposing
        Rf = denoise(self.orig_img,Tg)

        return Rf

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Low Light Image Enchancement')
    parser.add_argument('--img_path',type = str,default='./img1.png',help='Path to low light image to be enhanced')
    parser.add_argument('--save_img',type = str,default='./enhanced_img1.png',help='Save enhanced image as')
    parser.add_argument('--alpha',type = float,default= 0.15,help='Parameter for refining illumination')
    parser.add_argument('--gamma',type = float,default= 0.8,help='Gamma to use for gamma correction')
    args = parser.parse_args()
    enhancer = LLenhance(args)
    enhanced_img = enhancer.enhanceLL(args.alpha,args.gamma)
    cv2.imwrite(args.save_img, enhanced_img)