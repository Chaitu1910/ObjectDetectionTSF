# ObjectDetectionTSF

## Hello Everyone.

This is Chaitanya Kumar. This is a video of a completed Task 1 under the IoT and Compututer
Vision GRIP at The Sparks Foundation.

Task :- Implementation of Object Detector to identify classes of objects in an image.

The algorithm that has been used is based on deep learning. Here I have used SSD ( Single
Shot Multi Box Detector). The algorithm is deployed through the Caffe framework. The SSD 
is an efficient and quick approach to detect boundaries of different features in the input
image. The system uses convolutional features for achieving this. The algorithm for detecting
objects in an image are as follows:


1) Scale the input image, this is to ease the process of detection, as images are of different 
dimensions. This is done with calculations through the numpy library.

2) Initialization of labels. Prediction of network and Resizing.

3) Object detection and drawing of a box after detection.

4) Filter prediction and mapping of labels.

5) Prediction of Confidence 
