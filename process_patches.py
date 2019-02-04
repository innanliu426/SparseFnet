import scipy
import numpy as np
from matplotlib import pyplot as plt

###### cut whole pictures into small patches for NN ######
def patches(x, stepc, stepr, pixc, pixr):
    n1, n2 = x.shape
    l = []
    for i in range(0, n1-pixr, stepr):
        for j in range(0, n2-pixc, stepc):
            patch = x[i:i+pixr, j:j+pixc]
            l.append(patch)
    return l


###### 2D DCT transformation ######
def dct2(x):
    return scipy.fftpack.dct(scipy.fftpack.dct(x, norm = 'ortho' ,axis=0),  norm = 'ortho' ,axis=1)
def idct2(x):
    return scipy.fftpack.idct(scipy.fftpack.idct(x, norm = 'ortho' ,axis=1),  norm = 'ortho' ,axis=0)



###### pre-training and post-training DCT on patches ######
def transx(x):
    fftx=[]
    for i in range(len(x)):
        trans = dct2(x[i])
        fftx.append(trans)
    return fftx
def transy(y,output_r, output_c):
    ffty=[]
    for i in range(len(y)):
        trans = dct2(y[i])
        ffty.append(trans[:output_r, :output_c])
    return ffty
def transback(x):
    length = x.shape[0]
    for i in range(length):
        x[i,:,:,0] = idct2(x[i,:,:,0])
    return x



###### reshape data and transform patches ######
def dct_data(x_data,y_data):  
    #print(len(x_data), len(y_data))
    input_r, input_c = x_data[0].shape
    output_r, output_c = y_data[0].shape
    
    x_data = transx(x_data)
    y_data = transx(y_data)
    x_train = x_data
    y_train = y_data

    input_r, input_c = x_train[0].shape
    output_r, output_c = y_train[0].shape 

    x_train = np.reshape(x_train , (len(x_train), input_r,input_c, 1))  
    y_train = np.reshape(y_train , (len(y_train), output_r,output_c, 1))  

    return input_r, input_c,output_r, output_c,x_train, y_train



###### reconstruct recovered high-resolution patches ######
def reconst(pred, x_train, X, col_enlarge_factor, row_enlarge_factor, num_pic):
    l = int(x_train.shape[0]/num_pic)
    result = pred[:l,:,:,0]

    x_r = result.shape[1]
    x_c = result.shape[2]
    X_r = X.shape[0] *row_enlarge_factor 
    X_c = X.shape[1] *col_enlarge_factor
    recon = np.zeros((X_r, X_c))
    count = np.zeros((X_r, X_c))
    mark = np.ones((x_r,x_c))
    cstep = col_enlarge_factor
    rstep = row_enlarge_factor
    nc = int((X_c - x_c)/cstep)
    nr = int((X_r - x_r)/rstep)   

    k = 0
    for i in range(nr):
        for j in range(nc):
            recon[i*rstep:i*rstep+x_r,j*cstep:j*cstep+x_c] = recon[i*rstep:i*rstep+x_r,j*cstep:j*cstep+x_c] + result[k,:,:]
            count[i*rstep:i*rstep+x_r,j*cstep:j*cstep+x_c] = count[i*rstep:i*rstep+x_r,j*cstep:j*cstep+x_c] + mark
            k += 1
    recon = recon / count
    
    return recon


###### show the compressed, recovered and ground truth pictures, and find PSNR for reconstruction quality ######
def detail(x,y,z,i1,i2,j1,j2):
    fig = plt.figure(figsize=(16,5))
    plt.subplot(1, 3, 1)
    fig.gca().set_title('Original')
    imgplot =  plt.imshow(x[i1:i2,j1:j2], interpolation='nearest', aspect='auto',cmap = 'gray')
    plt.subplot(1, 3, 2)
    fig.gca().set_title('Reconstructed')
    imgplot =  plt.imshow(y[i1:i2,j1:j2], interpolation='nearest', aspect='auto',cmap = 'gray')
    plt.subplot(1, 3, 3)
    fig.gca().set_title('Ground Truth')
    imgplot =  plt.imshow(z[i1:i2,j1:j2], interpolation='nearest', aspect='auto',cmap = 'gray')
    plt.axis('off')
    plt.show()
    
    mse = np.mean((y[i1:i2,j1:j2]-z[i1:i2,j1:j2])**2)
    psnr = 20*np.log(255)-10*np.log(mse)
    print('PSNR:',  psnr)