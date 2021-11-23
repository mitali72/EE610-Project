import numpy as np
import bm3d
import cv2

def initial_map(img):
    """
    input - 
        img - M x N x 3 RGB image
    output - 
        imap - M x N x 3 illumination map
    """
    imap = np.amax(img, axis=2)
    imap = imap + (1e-6)*(imap==0)
    return imap


def gamma_correct(imap, gamma=0.8):
    """
    input - 
        imap - M x N illumination map
        gamma - float value for gamma
    output - 
        asdf - M x N gamma corrected map
    """
    gamma_imap = np.power(imap, gamma)

    return gamma_imap


def denoise(L, imap):
    """
    inputs - 
        L - M x N x 3 input image
        imap - M x N illumination map
    outputs - 
        Rf - M x N x 3 clear image
    """
    T = np.repeat(imap[:, :, np.newaxis], 3, axis=2)
    R = np.divide(L, T + 1e-5)

    # convert to yuv for denoising
    r = R[:,:,2]
    g = R[:,:,1]
    b = R[:,:,0]

    Y = 0.299*r + 0.587*g + 0.114*b
    U = -0.14713*r - 0.28886*g + 0.436*b
    V =0.615*r - 0.51499*g - 0.10001*b
    yd = bm3d.bm3d(Y, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)

    
    # convert back to bgr
    Rd = np.zeros((R.shape))
    r = yd + 1.13983*V
    g = yd - 0.39465*U - 0.5806*V
    b = yd + 2.03211*U

    Rd[:,:,0] = b
    Rd[:,:,1] = g
    Rd[:,:,2] = r

    #Recomposing
    Rf = L + np.multiply(Rd, (1-T))
    Rf= Rf*255
    Rf = np.clip(Rf, 0, 255)
    return Rf

def LOE(img, imgr):
    """
    inputs - 
        img - M x N x 3 - input image
        imgr - M x N x 3 - reconstructed image
    outputs - 
        loe - lightness order error
    """
    Q = np.amax(img, axis=2).reshape((img.shape[0]*img.shape[1]))
    Qr = np.amax(imgr, axis=2).reshape((img.shape[0]*img.shape[1]))
    loe = 0
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            Ul = 1 if Q[i]>=Q[j] else 0
            Ur = 1 if Qr[i]>=Qr[j] else 0
            tempsum = Ul^Ur
            loe += tempsum
    loe/=Q.shape[0]
    return loe
