##############################################################################
# CNNs in the Frequency Domain for Image Super-resolution
# by Liu, Paffenroth and Weiss
# innanliu426@gmail.com
# https://github.com/innanliu426/SparseFnet
##############################################################################


### define resolution settings 
### for measurement rate, i.e. row_enlarge_factor = col_enlarge_factor = 3: MR = 1/(3*3) = 0.1
row_enlarge_factor = 3
col_enlarge_factor = 3
input_pixr = 10  # number of row pixel for the patch
input_pixc = 10  # number of column pixel for the patch


from process_patches import *
import keras
import pandas as pd
from scipy import misc
from skimage import io
from skimage.transform import resize
from keras.layers import Input,Conv2D
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Reshape


####################### input data and pre-trianing process #######################
### read .png picture
def proc_data(file):
    img = io.imread('test_org_5.png', as_grey=True)
    ### resize ground truth to be 300x300
    truth = resize(img, (300,300), anti_aliasing=True) 

    ### compress the ground truth 
    x = resize(truth, (int(truth.shape[0]/row_enlarge_factor), int(truth.shape[1]/col_enlarge_factor)), anti_aliasing=True)
    x = scipy.ndimage.gaussian_filter(x,1)

    ### use grey scale from a RGB picture
    X = x[:,:]
    reduced = truth[:,:]
    
    ### cut the whole picture into small patches
    x_patches = patches(X, stepr = 1, stepc = 1, pixr = input_pixr, pixc = input_pixc)
    y_patches = patches(reduced, stepr = row_enlarge_factor, stepc = col_enlarge_factor,\
                              pixr = input_pixr*row_enlarge_factor, pixc = input_pixc*col_enlarge_factor)
    input_r, input_c,output_r, output_c,x_train, y_train = dct_data(x_patches,y_patches)  
    
    return X, reduced, input_r, input_c,output_r, output_c,x_train, y_train
### input .png file name here
X, reduced, input_r, input_c,output_r, output_c,x_train, y_train = proc_data(file = '***.png')


####################### the model #######################
def resolution(input_img):
      
    x = Flatten()(input_img)
    x = Dense(output_r*output_c,  activation = "linear", name='level1' )(x)
    x = Reshape((output_r, output_c, 1))(x)
    x = Conv2D(64, (9, 9), activation='relu', padding='same', name='level2')(x)
    x = Conv2D(1, (5, 5), padding='same', name='output')(x)

    out = Reshape((output_r, output_c,1))(x)
    return out

input_img = Input(shape = (input_r, input_c, 1))
resolution = Model(input_img, resolution(input_img))
resolution.compile(loss='mean_squared_error', optimizer = 'adam')
# resolution.summary()

####################### train #######################
resolution_train = resolution.fit(x_train, y_train, batch_size=128,epochs=100,verbose=1,)
plt.plot(resolution_train.history['loss'])
plt.ylim((0,0.05))
###save the model to .h5 file
resolution.save('trained_SparesFnet.h5') 


####################### show training result #######################
### reconstruct training patches
pred = transback(resolution.predict(x_train))
### reconstruct the whole picture
whole_pic = reconst(pred, x_train, X, col_enlarge_factor, row_enlarge_factor, num_pic = 1)
### show the compressed, recovered and ground truth pictures, and find PSNR for reconstruction quality
detail(X,whole_pic,reduced,0,reduced.shape[0]-5,0,reduced.shape[1]-5)