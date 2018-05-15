# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




[//]: # (Image References)

[center1]: ./images/center_2018_05_07_15_15_41_606.jpg "Center Driving 1"
[left1]: ./images/left_2018_05_07_15_15_41_606.jpg "Left Driving 1"
[right1]: ./images/right_2018_05_07_15_15_41_606.jpg "Right Driving 1"
[recovery1]: ./images/center_2018_05_08_08_53_46_767.jpg "Recovery from oversteer 1"
[recovery2]: ./images/center_2018_05_08_08_53_47_509.jpg "Recovery from oversteer 2"
[recovery3]: ./images/center_2018_05_08_08_53_48_330.jpg "Recovery from oversteer 3"
[flipped1]: ./images/center_2018_05_08_09_09_28_413.jpg "Flipped 1"
[flipped2]: ./images/center_2018_05_08_09_09_28_413_flipped.jpg "Flipped 2"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md | README.md summarizing the results
* video.mp4 | video_20.mp4 are videos showing the autonomous laps I achieved. The first is at the default 60 fps, while the second at 20fps. My poor integrated intel gfx produced only 7 images per sec.
* video_drop025.mp4. Please read the final section of this report.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
./drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.It does not contain comments to explain how it works, although this report references the respective code lines for every bit of functionality.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My Sequential model consists of the following layers:

* 1 Normalization as x / 127.5 - 1.0 (model.py:46)
* 1 Cropping at ((60, 25), (0, 0)) to remove unecessary clutter from the images which could bias the model  (model.py:47)
* 3 Convolutions with 5x5 filter sizes and depths 24, 36, 48 respectively, at a 2x2 stride and ReLU activation (model.py:48-50)
* 2 Convolutions with 3x3 filter sizes and depths 64 both, at a 1x1 stride and ReLU activation (model.py:51-52)
* Dropout to reduce overfitting (model.py:53)
* Flatten (model.py:54)
* 5 Fully Connected layers of 1152, 100, 50, 10, 1 neurons respectively (model.py:55-59)
* A summary node that produces the following table:

```
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 75, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 36, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 2, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1152)              4867200   
_________________________________________________________________
dense_2 (Dense)              (None, 100)               115300    
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 5,119,419
Trainable params: 5,119,419
Non-trainable params: 0
_________________________________________________________________
```

The model.py supports several parameter that override default values of hyperparameters. The final configuration used was

```sh
./model.py -k 0.75 -e 10 -b 120
```

where 
* -k: 0.75, is the dropout probability at 75%. Please read the note at the end.
* -e: 10 is the number of epochs as a large enough number to observe the point of overfitting, in combination with per-epoch checkpoints
* -b: 120 is the batch size

Other parameters where the default give the best results are:
* -t: 0.2 for the percentage of data for testing at 20%
* -l: 0.001 for the learning rate
* -c to automatically store only the best model as model.h5
* -p to plot the validation accuracy per epoch


#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer to reduce overfitting (model.py:53). The keep probability used was 75% but several values were used for experimentation in the range between 50% to 85%.

The model was trained and validated on different data sets to ensure that the model was not overfitting. This was achieved using a generator that shuffles the input per epoch (model.py:65-104).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py:61).


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The images were fed into the model in a sequence of 1. For the side cameras a small adjustment was made to the steering value at -0.2, +0.2 respectively (model.py:34-38).

For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model architecture was derived from the famous NVIDIA paper [1].

Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., … Zieba, K. (2016). End to End Learning for Self-Driving Cars, 1–9. Retrieved from http://arxiv.org/abs/1604.07316

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Immediately the model performed quite well, but produced a noticeable deviation in accuracy between training and validation loss. with the latter being higher. This led me to add adropout layer, my only change over the original NVIDIA model. The results were very positive.

Here are the actual results per epoch

* 321/321 [==============================] - 54s 169ms/step - loss: 0.0346 - val_loss: 0.0189
* 321/321 [==============================] - 54s 169ms/step - loss: 0.0188 - val_loss: 0.0162
* 321/321 [==============================] - 54s 168ms/step - loss: 0.0174 - val_loss: 0.0158
* 321/321 [==============================] - 54s 168ms/step - loss: 0.0168 - val_loss: 0.0155
* 321/321 [==============================] - 54s 167ms/step - loss: 0.0162 - val_loss: 0.0150
* 321/321 [==============================] - 54s 168ms/step - loss: 0.0159 - val_loss: 0.0151
* 321/321 [==============================] - 54s 169ms/step - loss: 0.0151 - val_loss: 0.0144
* 321/321 [==============================] - 54s 168ms/step - loss: 0.0146 - val_loss: 0.0139
* 321/321 [==============================] - 54s 168ms/step - loss: 0.0141 - val_loss: 0.0143
* 321/321 [==============================] - 54s 169ms/step - loss: 0.0136 - val_loss: 0.0141

During my experimentation it quickly became apparent that parameter tuning causes the best results at an arbitrary epoch number. Thus I implemented the checkpoint mechanism, saving a model after every epoch, allowing me to observe the full acuuracy curve (number 9 at the above results, when the val_loss raises again) thus always choosing the best model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track:

* Initially the first left curve before the bridge
* The bridge when the car would deviate to the side and get stuck driving onto the ledge of the bridge
* The subsequent left curve where a boundary on the right side is missing
* The only right curve right afterwards

These obstacles were addressed by improving the training set. For the right curve I produced data drving in the opposite direction. I also emulated situations when the car would correct it self back to the center from an over/under steered position.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road or even stepping on a lane.


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if it accidentaly over/under-steered itself to the edge. These images show what a recovery looks like starting from ... :

![alt text][recovery1]
![alt text][recovery2]
![alt text][recovery3]

To augment the data set, I also flipped images horizontally and reversed the steering; this would augment the dataset teaching the model to turn right more efficiently. For example, here is an image that has then been flipped:

![alt text][flipped1]
![alt text][flipped2]

After the collection process, I had 20706 number of original data points, 41412 total due to flipping. Next the data where normalized and cropped.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 6 and 8 depending on different hyper parameter values. I used an adam optimizer so that manually training the learning rate wasn't necessary.

A lesson from this project is that the data quality is probably the most important factor. I made several data collections before achievng the submitted results. Initially I tried with only keyboard on my laptop (intel gfx); the produced data were bad and the model was unable to maintain the car on the road. I then used a gamepad; it improved results dramatically with no modification to the model, producing the first full lap autonomously; nonetheless perforamnce was unreliable. Then I used a windows latop with the dedicated Geforce 970m and a gamepad; finally the car would drive around the track autonomously again with no modification to the model or the parameter values. However it would brake very often and drive at a overall very low speed (<5mph). The submitted lap is by using the provided training data from Udacity, again at no modification to the model or the parameters in any way.


## Dropout rate parameter in Keras

Throughout the curse material the dropout parameter is called ``keep_prob``, denoting the reverse of the Dropout function, the probability to keep an element. This notation is true for tensorflow. I believe we all agree it is counterintuitive.

However, in Keras the rate parameter is actully the probability to drop an element! This is not explicitely mentioned in the course material, or if it is is only a sidenote. Please emphasise it when introducing Keras and also have an emphatic note in text. In section 12.8. Dropout in Keras, this is written:
```Add a dropout layer after the pooling layer. Set the dropout rate to 50%. (Using Keras v1.2.1 from the starter kit? See this archived documentation about dropout.)```
Obviously giving an example of 0.5 is like wanting students to fall for this error! Did you? Did you actually do this on purpose? Some of us are professionals with work and family obligations and do not have the time to play games!

Finally, it is NOT my mistake for not reading the documentation. Of course I could have, but I'm not obligated to when I'm paying you to teach me something and I expect the material to bo correct and up to date. I do not expect to have to double check everything you say. You provide a conda environment with specific versions you have selected.

I actually spent a lot of effort working with suboptimal data that produced mediocre results, only because I was setting the ``drop`` probability to 0.75! Once I realized the mistake, I rerun my code with my own data and had immediately a perfect run over track 1. All in all I consider this omission on behalf of the Udacity crew an unacceptable mistake. I include file video_drop025.mp4 to showcase the results I would have had 2 weeks earlier than today!
