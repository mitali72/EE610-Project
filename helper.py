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
    # K = np.power(255, gamma)
    gamma_imap = np.power(imap, gamma)
    # gamma_imap = gamma_imap*255

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

    # let entries be in (0,1)
    L = L/255
    T = T/255

    R = np.divide(L, T + 1e-5)
    yuv_img = cv2.cvtColor(R, cv2.COLOR_BGR2YUV).astype(np.float32)
    yd = bm3d.bm3d(yuv_img[:,:,0], sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    Rd = np.array(yuv_img)
    Rd[:,:,0] = yd
    Rd = cv2.cvtColor(Rd, cv2.COLOR_YUV2BGR).astype(np.float32)
    #Recomposing
    Rf = L + np.multiply(Rd, (1-T))
    # Rf= Rf*255
    return Rf
