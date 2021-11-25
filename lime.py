from typing import DefaultDict
import cv2
import numpy as np
import argparse
from helper import *
import time

class LLenhance():

    def __init__(self,args):
        self.orig_img = cv2.imread(args.img_path)
        if self.orig_img is None:
            raise ValueError("Invalid image path")
        self.orig_img = self.orig_img/255

    def fwdDiffToep(self, m):
        """
        input -
            m - size of array
        output - 
            D - discrete derivative operator (forward difference)
        """
        D = np.zeros((m,m))
        for i in range(m-1):
            D[i,i] = -1
            D[i,i+1] = 1

        D[m-1][0] = 1
        D[m-1][m-1] = -1
        return D

    def gradT(self,T, Dv, Dh):
        """
        inputs - 
            T - illumination map
            Dv - vertical gradient filter
            Dh - horizontal gradient filter
        output -
            horiozntal gradient of T, vertical gradient of T
        """
        return np.matmul(T,Dh), np.matmul(Dv,T)

    def Tdenom(self, mu,m,n):
        """
        computing denominator of T_{t+1} in its closed form updation solution
        inputs - 
            mu - positive scalar associated with the Z term (Largangian multiplier)
            m,n - size of matrices
        output - 
            denom - denominator of T_{t+1}
        """
        Dh = np.zeros((m,n))
        Dv = np.zeros((m,n))
        Dh[1,1] = -1
        Dh[1,2] = 1
        Dv[1,1] = -1
        Dv[2,1] = 1
        Dhfft = np.fft.fft2(Dh)
        Dvfft = np.fft.fft2(Dv)
        denom = 2 + mu*(Dhfft*np.conjugate(Dhfft) + Dvfft*np.conjugate(Dvfft))
        return denom
    
    def refineT(self,Tinit,alpha = 0.15,rho = 1.1):
        """
        implementing exact solver to refine initial illumination map 
        inputs - 
            Tinit - M * N initial illumination map
            alpha - balance coefficient
            rho - coefficient for mu update
        outputs -
            T - final illumination map
        """
        #Objective: min||T−T||F +α||W◦∇T||1
        m,n = Tinit.shape[0],Tinit.shape[1]
        #Initialising G: gradT, Z: Lagrangian multiplier
        G = np.zeros((2*m,n))
        Z = np.zeros((2*m,n))

        #Discrete gradient operators with forward difference : f'(x) = f(x+1)-f(x) (first order gradient)
        #Vertical gradient filter
        Dv = self.fwdDiffToep(m)
        #Horizontal gradient filter
        Dh = self.fwdDiffToep(n).T
        #First order gradient of inital estimate of illumination map
        delTih,delTiv = self.gradT(Tinit, Dv, Dh)
        
        #Initialising weight matrix using strategy II
        Wh = 1/(np.abs(delTih)+1e-4)
        Wv = 1/(np.abs(delTiv)+1e-4)
        W = np.concatenate((Wh,Wv))

        mu = 0.05
        delta = 1e-5
        max_iter = 10000
        
        while(max_iter):
            #T sub-problem: T(t+1) <- F^-1(F(2*T^ + muD'(G-Z/mu))/(2+mu*sum(F(Dd)'F(Dd)) over {h,v}))
            x = G - Z/mu
            num = np.fft.fft2(2*Tinit + mu*(np.matmul(x[:m,:],Dh)+np.matmul(Dv,x[m:,:])))
            denom = self.Tdenom(mu,m,n)
            T = np.fft.ifft2(num/denom)
            T = np.real(T)

            #G sub-problem: G(t+1) = S_epsilon[x]; x = ∇T(t+1) + Z/mu , epsilon = alpha*W/mu
            #S_epsilon[x] = sgn(x)max(|x|-epsilon,0)
            delTh,delTv = self.gradT(T, Dv, Dh)
            delT = np.concatenate((delTh,delTv))
            epsilon = alpha*W/mu
            x = delT +Z/mu
            G = np.sign(x)*np.clip(np.abs(x)- epsilon,a_min = 0,a_max = None)

            #Updating Z and mu
            Z = Z + mu*(delT - G)
            mu = mu*rho

            #check convergence: ||∇T(t+1) − G(t+1)||F ≤ δ||Tˆ||F
            if(np.linalg.norm(delT-G,'fro')< delta*np.linalg.norm(Tinit,'fro')):
                break
            max_iter-=1

        print("Number of iterations: ", max_iter)
        return T

    def enhanceLL(self,alpha,gamma):
        '''
        input: 
            alpha - balance coefficient
            gamma - gamma correction coefficient
        output:
            Rf - Enhanced image
        '''
        #Step1: Estimate initial illumination map (T^(x) <- max(L(x)) over {R,G,B})
        Tinit = initial_map(self.orig_img)
        #Step2: Refine illumination map T based on T^ via exact solver
        T = self.refineT(Tinit,alpha,rho = 2.0)
        #Step3: Gamma correction on T via T <- T^gamma
        Tg = gamma_correct(T,gamma)
        #Step4: Denoising R using BM3D (Rd) and Recomposing (Rf)
        Rf = denoise(self.orig_img,Tg)

        return Rf

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Low Light Image Enchancement')
    parser.add_argument('--img_path',type = str,default='./img1.png',help='Path to low light image to be enhanced')
    parser.add_argument('--save_img',type = str,default='./enhanced_img1.png',help='Save enhanced image as')
    parser.add_argument('--alpha',type = float,default= 0.15,help='Parameter for refining illumination')
    parser.add_argument('--gamma',type = float,default= 0.8,help='Gamma to use for gamma correction')
    args = parser.parse_args()

    start = time.time()
    enhancer = LLenhance(args)
    enhanced_img = enhancer.enhanceLL(args.alpha,args.gamma)
    cv2.imwrite(args.save_img, enhanced_img)
    end = time.time()
    print("Runtime:", end - start)
