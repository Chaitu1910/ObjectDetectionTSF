import numpy as np
import argparse
import cv2 

parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network')
parser.add_argument("--image", default= "Person.jpg", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model'
                                       )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model'
                                      )
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Initialization of Labels
classNames = { 0: 'Background',
    1: 'Aeroplane', 2: 'Bicycle', 3: 'Bird', 4: 'Boat',
    5: 'Bottle', 6: 'Bus', 7: 'Car', 8: 'Cat', 9: 'Chair',
    10: 'Cow', 11: 'Diningtable', 12: 'Dog', 13: 'Horse',
    14: 'motorbike', 15: 'Person', 16: 'Pottedplant',
    17: 'Sheep', 18: 'Sofa', 19: 'Train', 20: 'TVMonitor' }

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
# Read image
frame = cv2.imread(args.image)
frame_resized = cv2.resize(frame,(300,300)) 
heightFactor = frame.shape[0]/300.0
widthFactor = frame.shape[1]/300.0 
# Scaling is performed to normalize the size of the object.
blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
#Set to network the input blob 
net.setInput(blob)
#Prediction of network
detections = net.forward()

frame_copy = frame.copy()
frame_copy2 = frame.copy()
cols = frame_resized.shape[1] 
rows = frame_resized.shape[0]


for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2] #Confidence of prediction 
    if confidence > args.thr: # Filter prediction 
        class_id = int(detections[0, 0, i, 1]) 

        # Object location 
        xLeftBottom = int(detections[0, 0, i, 3] * cols) 
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop   = int(detections[0, 0, i, 5] * cols)
        yRightTop   = int(detections[0, 0, i, 6] * rows)

        xLeftBottom_ = int(widthFactor * xLeftBottom) 
        yLeftBottom_ = int(heightFactor* yLeftBottom)
        xRightTop_   = int(widthFactor * xRightTop)
        yRightTop_   = int(heightFactor * yRightTop)
        # Draw location of object  
        cv2.rectangle(frame_resized, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                      (0, 255, 0))

        cv2.rectangle(frame_copy, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),
                      (0, 255, 0),-1)
opacity = 0.3
cv2.addWeighted(frame_copy, opacity, frame, 1 - opacity, 0, frame)

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2] #Confidence of prediction 
    if confidence > args.thr: # Filter prediction 
        class_id = int(detections[0, 0, i, 1]) # Class label

        xLeftBottom = int(detections[0, 0, i, 3] * cols) 
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop   = int(detections[0, 0, i, 5] * cols)
        yRightTop   = int(detections[0, 0, i, 6] * rows)

        xLeftBottom_ = int(widthFactor * xLeftBottom) 
        yLeftBottom_ = int(heightFactor* yLeftBottom)
        xRightTop_   = int(widthFactor * xRightTop)
        yRightTop_   = int(heightFactor * yRightTop)
        cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),
          (0, 0, 0),2)
        if class_id in classNames:
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.8, 1)

            yLeftBottom_ = max(yLeftBottom_, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_ - labelSize[1]),
                                 (xLeftBottom_ + labelSize[0], yLeftBottom_ + baseLine),
                                 (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom_, yLeftBottom_),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0))
            print (label)  
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
