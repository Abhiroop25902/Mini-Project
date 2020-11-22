# YOLO object detection
import cv2 as cv
import numpy as np
from scipy.spatial import distance

def object_detection_YOLO(img,threshold,nms_threshold):
    # determine the output layer
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the image
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)      #blob = boxes
    #blobs goes as the input to YOLO

    #inputting blob to the Neural Network
    net.setInput(blob)
    #t0 = time.time()
    outputs = net.forward(ln)   #finds output
    #t = time.time()
    #print('time=', t-t0)

    boxes = []
    confidences = []
    classIDs = []
    centers = []
    h, w = img.shape[:2]

    for output in outputs:  #Outputs have all the detection and their probability for every class
        for detection in output:    #detection is the the list of all probabilities with box dimension in start
            scores = detection[5:]  #everything in array after 5th elements
            classID = np.argmax(scores)     #picks the maximum probability
            confidence = scores[classID]    
            if (confidence > threshold) & (classID == 0):
                #first 4 elemensts are box characteristics normalized to range(0,1)
                #first two element are middle co-ordinate
                # next two are width and height of blob           
                box = detection[:4] * np.array([w, h, w, h])    
                (centerX, centerY, width, height) = box.astype("int")   #typecasting to int, as array indexes are int
                x = int(centerX - (width / 2))      #finding upper corner
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]   #changing origin to top left and typecasted to int
                boxes.append(box)                       #added the box to boxes
                confidences.append(float(confidence))   #added confidence to confidences
                classIDs.append(classID)                #added classId to classIds
                centers.append((centerX,centerY))

    indices = cv.dnn.NMSBoxes(boxes, confidences,score_threshold=threshold,nms_threshold=nms_threshold)
    #score_threshold -> threshold for confidence
    #nms_threshold -> threshold for how close to blobs are, if two blobs are too close, one of them is discarded
    #closeness is determined by IoU (intersetction over Union)
    #discarding is based on confidence, higer confidence is retained

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2],boxes[i][3])
            r = (confidences[i], (w,h),(x, y),centers[i])
            results.append(r)

    return results

cap = cv.VideoCapture("pedestrians.mp4")
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv.CAP_PROP_BUFFERSIZE,10)


# Load names of classes and get random colors
classes = open('coco.names').read().strip().split('\n')
green = (0,255,0)
red = (0,0,255)
# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')     #Reads Network from .cfg and .weights
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)     #this sprcifies what type of hardware to use (GPU or CPU)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)         #sets preferable hardware


while True:

    ret,img = cap.read()

    if not ret:
        break

    threshold = 0.5
    nms_threshold = 0.4
    
    results = object_detection_YOLO(img,threshold,nms_threshold)
    #results = (confidence,dimension,top_left,ceters)
    #let n be the number of detected objects
    #confidence -> (n-dimension) confidence of the detected object
    #dimension = (width , height) of the box
    #top_left = (x,y) coordinate of top left corner
    #centers -> (nx2 dimension)the center of the box
    
    no_of_pixel_5m = 1000 #arbiitrary
    threshold_distance_pixel = 1 #arbitratry pixel value

    if len(results) > 2:    #showing output
        centers = np.array([r[3] for r in results])
        pairwase_distance = distance.cdist(centers,centers)
        violate = set()

        for i in range(0,pairwase_distance.shape[0]):
            for j in range(0, pairwase_distance.shape[1]):
                if pairwase_distance[i,j] < threshold_distance_pixel:
                    violate.add(i)
                    violate.add(j)

        for (i, (prob,box_dim,bbox,center)) in enumerate(results): 
            (x,y) = (bbox[0], bbox[1])     
            (w,h) = box_dim
            dist = w*h*5/no_of_pixel_5m

            color = green

            if i in violate:
                color = red

            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)     #making rectangle takes two opposite corners as input
            text = "{:.2f}m".format(dist)
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        
                
               

    cv.imshow('Video', img)
    #cv.imwrite("output.jpg",img)
    cv.waitKey(1)

cv.destroyAllWindows()
