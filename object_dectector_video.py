# YOLO object detection
import cv2 as cv
import numpy as np
from scipy.spatial import distance


def object_detection_YOLO(img, threshold, nms_threshold):
    results=[]
    # determine the output layers
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the image
    # blob is just a preprocessed image
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)  # blob = boxes
    # blobs goes as the input to YOLO
    # inputting blob to the Neural Network
    net.setInput(blob)
    # t0 = time.time()
    outputs = net.forward(ln)  # finds output
    # t = time.time()
    # print('time=', t-t0)

    boxes = []
    confidences = []
    centroids=[]
    h, w = img.shape[:2]
    for output in outputs:  # Outputs have all the detection and their probability for every class
        for detection in output:  # detection is the the list of all probabilities with box dimension in start
            scores = detection[5:]  # everything in array after 5th element
            classID = np.argmax(scores)  # picks the maximum probability
            confidence = scores[classID]

            if confidence > threshold and classID == 0:
                # first 4 elements are box characteristics normalized to range(0,1)
                # first two element are middle co-ordinate
                # next two are width and height of blob
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")  # typecasting to int, as array indexes are int
                x = int(centerX - (width / 2))  # finding upper left corner
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]  # changing origin to top left and typecasted to int
                boxes.append(box)  # added the box to boxes
                confidences.append(float(confidence))  # added confidence to confidences
                centroids.append((centerX,centerY))

    indices = cv.dnn.NMSBoxes(boxes, confidences, score_threshold=threshold, nms_threshold=nms_threshold)
    if len(indices):
        for i in indices.flatten():
            x,y=boxes[i][0],boxes[i][1]
            w,h=boxes[i][2],boxes[i][3]

            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # score_threshold -> threshold for confidence
    # nms_threshold -> threshold for how close to blobs are, if two blobs are too close, one of them is discarded
    # closeness is determined by IoU (intersection over Union)
    # discarding is based on confidence, higher confidence is retained

    return results


# def birds_eye_view(corner_points,width,height,image):
#     """
#     Compute the transformation matrix
#     @ corner_points : 4 corner points selected from the image
#     @ height, width : size of the image
#     return : transformation matrix and the transformed image
#     """
#     # Create an array out of the 4 corner points
#     corner_points = np.float32(corner_points)
#     # Create an array with the parameters (the dimensions) required to build the matrix
#     img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
#     # Compute and return the transformation matrix
#     matrix = cv.getPerspectiveTransform(corner_points,img_params)
#     img_transformed = cv.warpPerspective(image,matrix,(width,height))
#     return matrix,img_transformed

# def birds_eye_point(matrix,list_downoids):
#     """ Apply the perspective transformation to every ground point which have been detected on the main frame.
#     @ matrix : the 3x3 matrix
#     @ list_downoids : list that contains the points to transform
#     return : list containing all the new points
#     """
#     # Compute the new coordinates of our points
#     list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
#     transformed_points = cv.perspectiveTransform(list_points_to_detect, matrix)
#     # Loop over the points and add them to the list that will be returned
#     transformed_points_list = list()
#     for i in range(0,transformed_points.shape[0]):
#         transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
#     return transformed_points_list

cap = cv.VideoCapture("pedestrians.mp4")

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_BUFFERSIZE, 10)

# Load names of classes and get random colors
with open("coco.names") as f:
    classes = f.read().strip().split("\n")
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')  # gives different color to different classes

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')  # Reads Network from .cfg and .weights
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)  # this specifies what type of hardware to use (GPU or CPU)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)  # sets preferable hardware

threshold = 0.5
nms_threshold = 0.4
distance_px=100   # arbitrary value for now but looked best
writer= cv.VideoWriter("result.avi",cv.VideoWriter_fourcc(*"XVID"), 25,(1280, 720), True)
while True:
    ret, img = cap.read()
    if not ret: break

    results=object_detection_YOLO(img, threshold, nms_threshold)
    violate=set()

    if len(results)>1:
        centres=np.array([r[2] for r in results])
        D=distance.cdist(centres,centres,metric="euclidean")
        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < distance_px:
                    # update our violation set with the indexes of
                    # the centroid pairs
                    violate.add(i)
                    violate.add(j)
    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        # if the index pair exists within the violation set, then
        # update the color
        if i in violate:
            color = (0, 0, 255)
        # draw (1) a bounding box around the person and (2) the
        # centroid coordinates of the person,
        cv.rectangle(img, (startX, startY), (endX, endY), color, 2)
        cv.circle(img, (cX, cY), 5, color, 1)
    # draw the total number of social distancing violations on the
    # output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv.putText(img, text, (10, img.shape[0] - 25), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    cv.imshow('Video', img)

    # create an output file named result.avi (uncomment if wanted)
    # writer.write(img)
    k = cv.waitKey(1)
    if k == 27: break

writer.release()
cv.destroyAllWindows()
