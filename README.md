# Behavioral-Cloning
## Udacity Self Driving Car Engineer Nanodegree - Project 3
The goal of this project is to train a vehicle to drive autonomously in a simulator by taking camera images as inputs to a deep neural network and outputting a predicted steering angle based on the image. 

The project is broken up in to 3 distinct stages:
- Stage 1: Data Collection
- Stage 2: Image Preprocessing and Augmentation
- Stage 3: Building and Training a Neural Network
- Stage 4: Testing Model in Simulator

## Data Collection
In training mode in the simulator, data is collected by recording images through three forward facing cameras cameras mounted on the vehicle; one in the center, one on the left side and one on the right side. For each set of 3 images which is recorded, the simulator also outputs an associated steering angle which is used as a label for the images. 

| Left          | Center        | Right  |
| ------------- |:-------------:| ------|
|![Left] (https://github.com/JustinHeaton/Behavioral-Cloning/blob/master/Images/Left.jpg) | ![Center] (https://github.com/JustinHeaton/Behavioral-Cloning/blob/master/Images/Center.jpg) | ![Right] (https://github.com/JustinHeaton/Behavioral-Cloning/blob/master/Images/Right.jpg)

For training the model I recorded images for two different types of driving, **Controlled Driving** and **Recovery Driving**. 

| Controlled  Driving  | 
| -------------------- | 
|![Controlled] (https://github.com/JustinHeaton/Behavioral-Cloning/blob/master/Images/Controlled.gif) | 

In **controlled driving** I tried to maintain the vehicle close to the center of the driving lane as it travelled through the simulator course. For the **controlled driving** part of the dataset I used all three images (center, right, left) and I manufactured appropriate labels for the side camera images by adding a small amount of right turn to the left image labels and a small amount of left turn to the right image labels. The result of this is that when the car begins to drift from the center of the lane and finds itself in the approximate spot where the side camera had recorded the image, it will automatically know to steer itself back towards the center. 

Next there is **recovery driving**. In this image set I recorded driving segments which started from the outside of the lane (on either side) and recorded the vehicle driving back towards the center of the lane. Inevitably there will be times when the car drifts beyond the images recorded from the side cameras in **controlled driving** and it must make a larger steering  correction back towards center. The recovery images trained the car to make this correction when it finds itself drifting towards the curb/lane lines on either side of the road. Only the center camera images were used to train for recovery. 

| Recovery  Driving  | 
|:------------------:| 
|![Recovery] (https://github.com/JustinHeaton/Behavioral-Cloning/blob/master/Images/Recovery.gif) |

## Image Preprocessing and Augmentation
The images outputted by the simulator are 160 by 320 pixels with 3 color channels (RGB). The first step I took in preprocessing the images was to reduce the size so that I would be able to hold more images in memory at one time, and to speed up the training which would be extremely slow on full sized images. I chose to resize the images to 32 by 64 and to crop off 12 pixels from the top of each image because the top ~40% contains only blue sky and background objects which are not useful in training driving behavior. The final size of each image, and the input shape for my neural network, is 20 by 64 with 3 color channels.

Next I created a mirror image dataset which contains a copy of each image flipped vertically, along with mirrored labels with the direction of steering reversed for each value. The car in the simulator drives around a track in the counterclockwise, resulting in a majority of steering angles being in the left direction. I chose to make the mirror dataset to combat a bias towards left turns which appeared to be affecting the car even when driving on straight portions of road. The mirror dataset/labels was concatenated with the regular dataset/labels to make the final dataset (containing over 60,000 images) which was used in training the neural network.

The last preprocessing step was to create a validation set made up of 5% of the images/labels, randomly selected from the complete dataset. The validation set was used to monitor the performance of the neural network in training, but the true worth of the model would be determined in testing which is done in autonomous driving in the simulator. 

## Building and Training a Neural Network
For this problem I chose to use a convolutional neural network because  this is an image classification problem and I wanted to maintain the spatial structure of the images, and the relationships between adjacent pixels. Convolutional neural networks avoid the problem of having to flatten the images in to a 1 dimensional vector and are therefore more powerful in recongnizing 2 dimensional images.

To come up with a model architecture, I took inspiration from 
[this paper from Nvidia] (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) where they trained a convolutional neural network for a similar type of problem. Because I was beginning with a much smaller input shape I had to adjust my layer sizes compared to theirs but I chose to maintain 5 convolutional layers of increasing depth, just as they have done. I decided to implement one fewer fully connected layer than they did, but it still follows the same general pattern with decreasing number of neurons in each dense layer with a single neuron output layer at the top. My model also introduces a dropout layer after the first fully connected layer which helps to prevent overfitting to the training data. 

### The final architecture of my model is as follows:
- **Batch Normalization** (input shape = (20, 64, 3))
- **2D Convolutional Layer 1**: (depth = 16, kernel = 3 x 3, stride = 2 X 2, border mode = valid, activation = ReLu, output shape = (None, 9, 31, 16))
- **2D Convolutional Layer 2**: (depth = 24, kernel = 3 x 3, stride = 1 X 2, border mode = valid, activation = ReLu, output shape = (None, 7, 15, 24))
- **2D Convolutional Layer 3**: (depth = 36, kernel = 3 x 3, border mode = valid, activation = ReLu, output shape = (None, 5, 13, 36))
- **2D Convolutional Layer 4**: (depth = 48, kernel = 2 x 2, border mode = valid, activation = ReLu, output shape = (None, 4, 12, 48))
- **2D Convolutional Layer 5**: (depth = 48, kernel = 2 x 2, border mode = valid, activation = ReLu, output shape = (None, 3, 11, 48))
- **Flatten Layer**
- **Dense Layer 1**: (512 neurons)
- **Dropout Layer**: (keep prob = .5)
- **Relu Activation**
- **Dense Layer 2**: (10 neurons)
- **Relu Activation**
- **Output Layer** 

The model was compiled with an adam optimizer (learning rate = .0001), Mean Squared Error (mse) as a loss metric, and was set to train for 20 epochs with an early stopping callback which discontinues training when validation loss fails to improve for consecutive epochs. I chose to lower the learning rate in the adam optimizer because I found that the lower learning rate led to increased performance both in terms of mse and autonomous driving. When training is complete the model and weights are saved to be used for autonomous driving in the simulator.

## Testing model in the sumulator
The script drive.py takes in a constant stream of images, resizes and crops them to the input shape for the model, and passes the transformed image array to the model which predicts an appropriate steering angle based on the image. The steering angle is then passed to the car as a control and the car steers itself accordingly. The autonomous vehicle travels like this through the course, constantly putting out images and receiving steering angles and hopefully the model has been trained well enough that the steering angles it receives keep the vehicle driving safely in the middle of the lane and not drifting off the road or doing anything else which would be considered dangerous. 

The data collection and preprocessing techniques and model architecture outlined above were sufficient to build a model which drives safely around the course for multiple laps without hitting the curbs or drifting off of the road.
