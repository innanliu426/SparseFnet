# SparseFnet
paper: CNNs in the Frequency Domain for Image Super-resolution

This paper develops methods for recovering high-resolution images from low-resolution images by combining ideas inspired by compressive sensing techniques with super-resolution neural networks. Compressive sensing and convolutional neural network (CNN) are both popular techniques for image super-resolution. Compressive sensing leverages the existence of bases in which signals can be sparsely represented, and herein we use such ideas to improve the performance of super-resolution CNNs. In particular, we propose an improved model in which CNNs are used for super-resolution in the frequency domain. We demonstrate that the frequency domain, which provides a sparse representation of many natural images, helps to improve the performance of image super-resolution neural networks. In addition, we indicate that instead of numerous deep layers, a shallower architecture, in the frequency domain, is sufficient for classes of image super-resolution problems. 


## Model
![](https://github.com/innanliu426/SparseFnet/blob/master/framework.PNG | width=250)

### Train and Test
SparseFnet_train.py

SparseFnet_test.py

Use process_patches.py and .png picture 


### Requirements of Python environment 
numpy

scipy

matplotlib

keras

pandas

skimage
   
   
