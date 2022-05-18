# Hand gesture recognition using neural networks

## Developers

Mamatha Shetty

Kolla Neeraja

## Problem Statement

Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote.

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

- Thumbs up:  Increase the volume
- Thumbs down: Decrease the volume
- Left swipe: 'Jump' backwards 10 seconds
- Right swipe: 'Jump' forward 10 seconds  
- Stop: Pause the movie

Each video is a sequence of 30 frames (or images)

## Understanding the Dataset

The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use.

The data is in a [zip]([https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL]) file. The zip file contains a 'train' and a 'val' folder with two CSV files for the two folders.

<img width="403" alt="image" src="https://user-images.githubusercontent.com/16831674/169050875-d7417a0e-bffa-41a3-9808-17129a51f235.png">


These folders are in turn divided into subfolders where each subfolder represents a video of a particular gesture.

<img width="1060" alt="image" src="https://user-images.githubusercontent.com/16831674/169051194-f667c4a0-4ca5-40f9-b75f-f5bdbe782042.png">

Each subfolder, i.e. a video, contains 30 frames (or images). 

- Thumbs Up
  
<img width="1200" alt="image" src="https://user-images.githubusercontent.com/16831674/169051551-c300c816-eb91-4c19-9917-9480041e477b.png">

Note that all images in a particular video subfolder have the same dimensions but different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used to record the videos).

## Two Architectures: 3D Convs and CNN-RNN Stack

After understanding and acquiring the dataset, the next step is to try out different architectures to solve this problem. 

For analysing videos using neural networks, two types of architectures are used commonly. 

One is the standard **CNN + RNN** architecture in which you pass the images of a video through a CNN which extracts a feature vector for each image, and then pass the sequence of these feature vectors through an RNN. 

*Note:*
 - You can use transfer learning in the 2D CNN layer rather than training your own CNN 
 - GRU (Gated Recurrent Unit) or LSTM (Long Short Term Memory) can be used for the RNN

The other popular architecture used to process videos is a natural extension of CNNs - a **3D convolutional network**. In this project, we will try both these architectures.

## Data Preprocessing

We can apply several of the image procesing techniques for each of image in the frame.

### Resize

 We will convert each image of the train and test set into a matrix of size 120*120

### Cropping

Given that one of the data set is of rectangualr shape, we will crop that image to 120*120, this is different to resize, while resize changes the aspect ratio of rectangular image. In cropping we will center crop the image to retain the middle of the frame.


### Edge Detection
We will also experiemnt with edge detection for image processing

#### Sobel Edge Detection
Sobel edge detector is a gradient based method based on the first order derivatives. It calculates the first derivatives of the image separately for the X and Y axes.

https://en.wikipedia.org/wiki/Sobel_operator

#### Laplacian Edge Detection
Unlike the Sobel edge detector, the Laplacian edge detector uses only one kernel. It calculates second order derivatives in a single pass.

We will perform edge detection on each channel and use the comibined 3 channel as input 

### Normalization

We will use mean normaliztion for each of the channel in the image.

## Data Agumentation

We have a total of 600+ for test set and 100 sampels for validation set. We will increase this 2 fold by usign a simple agumentiaton technique of affine transforamtion.

### Affine Transformation

In affine transformation, all parallel lines in the original image will still be parallel in the output image. To find the transformation matrix, we need three points from input image and their corresponding locations in output image. Then cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.

Check below example

``` python
img = cv2.imread('drawing.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```

We will perform a same random affine transform for all the images in the frameset. This way we are generating new dataset from existing dataset.

## Generators

**Understanding Generators**: As you already know, in most deep learning projects you need to feed data to the model in batches. This is done using the concept of generators. 

Creating data generators is probably the most important part of building a training pipeline. Although libraries such as Keras provide builtin generator functionalities, they are often restricted in scope and you have to write your own generators from scratch. In this project we will implement our own cutom generator, our generator will feed batches of videos, not images. 

Let's take an example, assume we have 23 samples and we pick batch size as 10.

In this case there will be 2 complete batches of ten each
- Batch 1: 10
- Batch 2: 10
- Batch 3: 5

The final run will be for the remaining batch that was not part of the the full batch. 

Full batches are covered as part of the for loop the remainder are covered post the for loop.

Note: this also covers the case, where in batch size is day 30 and we have only 23 samples. In this case there will be only one single batch with 23 samples.

## Reading Video as Frames

Note that in our project, each gesture is a broken into indivdual frame. Each gesture consists of 30 individual frames. While loading this data via the generator there is need to sort the frames if we want to maintain the temporal information.

The order of the images loaded might be random and so it is necessary to apply sort on the list of files before reading each frame.


# Implementation 

## 3D Convolutional Network, or Conv3D

Now, lets implement a 3D convolutional Neural network on this dataset. To use 2D convolutions, we first convert every image into a 3D shape : width, height, channels. Channels represents the slices of Red, Green, and Blue layers. So it is set as 3. In the similar manner, we will convert the input dataset into 4D shape in order to use 3D convolution for : length, breadth, height, channel (r/g/b).

*Note:* even though the input images are rgb (3 channel), we will perform image processing on each frame and the end individual frame will be grayscale (1 channel) for some models

Lets create the model architecture. The architecture is described below:

While we tried with multiple ***filter size***, bigger filter size is resource intensive and we have done most experiment with 3*3 filter

We have used **Adam** optimizer with its default settings.
We have additionally used the ReduceLROnPlateau to reduce our learning alpha after 2 epoch on the result plateauing.


## Model #1

Build a 3D convolutional network, based loosely on C3D.

https://arxiv.org/pdf/1412.0767.pdf

```python

       model_a = Sequential()

      model_a.add(Conv3D(8,kernel_size=(3,3,3), input_shape=(30, 120, 120, 3),padding='same'))
      model_a.add(BatchNormalization())
      model_a.add(Activation('relu'))

      model_a.add(MaxPooling3D(pool_size=(2,2,2)))

      model_a.add(Conv3D(16, kernel_size=(3,3,3), padding='same'))
      model_a.add(BatchNormalization())
      model_a.add(Activation('relu'))

      model_a.add(MaxPooling3D(pool_size=(2,2,2)))

      model_a.add(Conv3D(32,kernel_size=(1,3,3), padding='same'))
      model_a.add(BatchNormalization())
      model_a.add(Activation('relu'))

      model_a.add(MaxPooling3D(pool_size=(2,2,2)))

      model_a.add(Conv3D(64, kernel_size=(1,3,3), padding='same'))
      model_a.add(BatchNormalization())
      model_a.add(Activation('relu'))

      model_a.add(MaxPooling3D(pool_size=(2,2,2)))

      #Flatten Layers
      model_a.add(Flatten())

      model_a.add(Dense(1000, activation='relu'))
      model_a.add(Dropout(0.5))

      model_a.add(Dense(500, activation='relu'))
      model_a.add(Dropout(0.5))

      #softmax layer
      model_a.add(Dense(5, activation='softmax'))

```
Model Summary

<img width="389" alt="image" src="https://user-images.githubusercontent.com/16831674/169067597-74eddcb5-01ed-49a5-aa3b-4cfe3afad45c.png">


## Model #2

Build a 3D convolutional network, aka C3D.

https://arxiv.org/pdf/1412.0767.pdf
```python
        model_b = Sequential()
model_b.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(nb_frames,nb_rows,nb_cols,nb_channel), padding='same'))
model_b.add(Activation('relu'))
model_b.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
model_b.add(Activation('relu'))
model_b.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
model_b.add(Dropout(0.25))

model_b.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
model_b.add(Activation('relu'))
model_b.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
model_b.add(Activation('relu'))
model_b.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
model_b.add(Dropout(0.25))

model_b.add(Flatten())
model_b.add(Dense(512, activation='relu'))
model_b.add(Dropout(0.5))
model_b.add(Dense(nb_classes, activation='softmax'))
```
Model Summary

<img width="370" alt="image" src="https://user-images.githubusercontent.com/16831674/169068247-a42e65bc-b1b5-4af3-acc0-7e47a59ee473.png">


## Model #3

<img width="370" alt="image" src="https://user-images.githubusercontent.com/16831674/169069781-b6186ca2-addd-40a4-9986-2571af409271.png">


## Model #4
<img width="389" alt="image" src="https://user-images.githubusercontent.com/16831674/169070487-eea58697-600e-455a-b15c-f3c42549924a.png">

## Model #5

<img width="371" alt="image" src="https://user-images.githubusercontent.com/16831674/169072116-95cc10ff-7bc9-4e65-89e3-713e90128e74.png">

## Model 6
<img width="387" alt="image" src="https://user-images.githubusercontent.com/16831674/169074350-4bec9b11-18ae-4836-8df3-7e661be69d5e.png">

## Model 7

Input and Output layers:

- One Input layer with dimentions 30, 120, 120, 3
- Output layer with dimentions 5

Convolutions :

- Apply 4 Convolutional layer with increasing order of filter size (standard size : 8, 16, 32, 64) and fixed kernel size = (3, 3, 3)
- Apply 4 Max Pooling layers, after each convolutional layer.

MLP (Multi Layer Perceptron) architecture:

- Batch normalization on convolutiona architecture
- Dense layers with 2 layers followed by dropout(0.3) to avoid overfitting

```python
        model_g = Sequential()

    model_g.add(Conv3D(8, 
                     kernel_size=(3,3,3), 
                     input_shape=input_shape,
                     padding='same'))
    model_g.add(BatchNormalization())
    model_g.add(Activation('relu'))

    model_g.add(MaxPooling3D(pool_size=(2,2,2)))

    model_g.add(Conv3D(16, 
                     kernel_size=(3,3,3), 
                     padding='same'))
    model_g.add(BatchNormalization())
    model_g.add(Activation('relu'))

    model_g.add(MaxPooling3D(pool_size=(2,2,2)))

    model_g.add(Conv3D(32, 
                     kernel_size=(1,3,3), 
                     padding='same'))
    model_g.add(BatchNormalization())
    model_g.add(Activation('relu'))

    model_g.add(MaxPooling3D(pool_size=(2,2,2)))

    model_g.add(Conv3D(64, 
                     kernel_size=(1,3,3), 
                     padding='same'))
    model_g.add(BatchNormalization())
    model_g.add(Activation('relu'))

    model_g.add(MaxPooling3D(pool_size=(2,2,2)))

    #Flatten Layers
    model_g.add(Flatten())

    model_g.add(Dense(nb_dense[0], activation='relu'))
    model_g.add(Dropout(0.3))

    model_g.add(Dense(nb_dense[1], activation='relu'))
    model_g.add(Dropout(0.3))

    #softmax layer
    model_g.add(Dense(nb_dense[2], activation='softmax'))

```

Model Summary

<img width="361" alt="image" src="https://user-images.githubusercontent.com/16831674/169080538-7b7a8e29-20d4-4b59-9b33-887cc979645f.png">


Model 7 gave us **test accuracy of 82% and validation accuracy of 82%** using all the 30 frames. The same model is submitted for the review. 
While we did try model lesser frames by using even frames but we felt more comfortable using full frame. Cropping and other preprocessing also did not 
affect much on the final accuracy.

## Model 8 

We tried one more model with changing the dropout value and keeping other parameters same as Model 7. Though it increased the accuracy of Training set to 89%, validation accuracy dropped to 80%. 

So Final Model is taken as Model 7.

Link to h5 file for Model 7: https://drive.google.com/drive/folders/1bT9Uenpmot3IspQlF0Fd4XmSlZRBltXG?usp=sharing


