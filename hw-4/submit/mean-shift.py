import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    return np.linalg.norm(x-X, axis=1)

def gaussian(dist, bandwidth):
    b = bandwidth
    d2 = dist*dist
    m = np.max(d2)
    r = np.exp(-0.5*(d2-m) / (b*b)) / np.sum(np.exp(-0.5*(d2-m) / (b*b))) 
    return r

def update_point(weight, X):
    return np.sum(X*weight, axis=0)

def meanshift_step(X, bandwidth=2.5):
    # calculate mean for all x
    # clist = []
    X = X.copy() # farkli bandwidth lerde loop da denedigim icin, bozmamak icin
    for i in range(len(X)):
        x = X[i]
        dist = distance(x,X)
        w = gaussian(dist, bandwidth=bandwidth)
        w = w.reshape(-1,1)
        c = update_point(w, X)
        X[i] = c
        # clist.append(c)
    # X = np.array(clist)
    return X
    # raise NotImplementedError('meanshift_step function not implemented!')

def meanshift(X):
    for _ in range(20):
        X = meanshift_step(X)
    return X

scale = 0.5    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab)
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
