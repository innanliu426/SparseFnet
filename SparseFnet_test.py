
row_enlarge_factor = 3
col_enlarge_factor = 3
input_pixr = 10  # number of row pixel for the patch
input_pixc = 10  # number of column pixel for the patch


from process_patches import *
import keras
import pandas as pd
import tensorflow as tf
from skimage import io
from scipy import misc
from skimage.transform import resize
from keras.layers import Input,Conv2D
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Reshape
from keras.models import load_model


resolution = load_model('trained_SparseFnet.h5',custom_objects={'tf':tf})


def proc_data(file):
    img = io.imread('test_org_5.png', as_grey=True)
    ### resize ground truth to be 300x300
    truth = resize(img, (300,300), anti_aliasing=True) 

    ### compress the ground truth 
    x = resize(truth, (int(truth.shape[0]/row_enlarge_factor), int(truth.shape[1]/col_enlarge_factor)), anti_aliasing=True)
    x = scipy.ndimage.gaussian_filter(x,1)

    ### use grey scale from a RGB picture
    X = x[:,:,0]
    reduced = truth[:,:,0]

    ### cut the whole picture into small patches
    x_patches = patches(X, stepr = 1, stepc = 1, pixr = input_pixr, pixc = input_pixc)
    y_patches = patches(reduced, stepr = row_enlarge_factor, stepc = col_enlarge_factor,\
                              pixr = input_pixr*row_enlarge_factor, pixc = input_pixc*col_enlarge_factor)
    input_r, input_c,output_r, output_c,x_train, y_train = dct_data(x_patches,y_patches)  
    
    return X, reduced, input_r, input_c,output_r, output_c,x_train, y_train


X, reduced, input_r, input_c,output_r, output_c,x_test, y_test = proc_data(file = 'test_org_9.png')
### reconstruct training patches
pred = transback(resolution.predict(x_test))
### reconstruct the whole picture
whole_pic = reconst(pred, x_test, X, col_enlarge_factor, row_enlarge_factor, num_pic = 1)
### show the compressed, recovered and ground truth pictures, and find PSNR for reconstruction quality
detail(X,whole_pic,reduced,0,reduced.shape[0]-5,0,reduced.shape[1]-5)