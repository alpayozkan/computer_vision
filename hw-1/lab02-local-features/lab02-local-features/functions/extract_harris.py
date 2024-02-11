import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter
import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # blur img => noise reduction
    I = img
    # I = cv2.GaussianBlur(img, (3,3), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    
    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    
    # gradient kernel - x direction
    k_x = np.array([
            [0, 0, 0],
            [-0.5, 0, +0.5],
            [0, 0, 0]
        ])
    Ix = signal.convolve2d(I, k_x, boundary='symm', mode='same')

    # gradient kernel - y direction
    k_y = np.array([
        [0, -0.5, 0],
        [0, 0, 0],
        [0, +0.5, 0]
    ])
    Iy = signal.convolve2d(I, k_y, boundary='symm', mode='same')

    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    
    # apply noise reduction => derivatives
    # Ixs = cv2.GaussianBlur(Ix, (3,3), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    # Iys = cv2.GaussianBlur(Iy, (3,3), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    Ixs = Ix
    Iys = Iy
    
    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    
    # matrix elements calculation
    Ixx = Ixs*Ixs
    Ixy = Ixs*Iys
    Iyx = Ixy
    Iyy = Iys*Iys 

    # also apply gaussian weighting => matrix elements as in the sum[w*[[Ixx, Ixy],[Iyx, Iyy]]]
    Ixxs = cv2.GaussianBlur(Ixx, (3,3), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    Ixys = cv2.GaussianBlur(Ixy, (3,3), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    Iyxs = cv2.GaussianBlur(Iyx, (3,3), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    Iyys = cv2.GaussianBlur(Iyy, (3,3), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

    M = \
    np.array([
        [Ixxs, Ixys],
        [Iyxs, Iyys]
    ])
    # arrange dimensionality to broadcast for trace and determinant calculation
    MM = np.transpose(M, (2,3,0,1)) # (h,w, 2,2)

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    
    det = np.linalg.det(MM) # (h,w)
    tr = MM.trace(axis1=2, axis2=3) # (h,w)
    R = det - k*(tr**2)
    C = R
    
    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format

    # non-max suppression
    # suppress local weak signals
    mask = (ndimage.maximum_filter(R, 3) == R)
    R_sup = mask*R
    # if below threshold leave out
    R_filt = (R_sup>thresh)*R_sup
    RR = R_filt

    # select non-zero corners
    cc = np.where(RR>0)
    corners = np.stack([cc[1], cc[0]]) # (q,2), need to change x,y position => careful!!!
    corners = np.transpose(corners, (1,0)) # (2,q) => (q,2)

    return corners, C

