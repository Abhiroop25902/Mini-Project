import cv2 as cv

def total_frames(file_name):
    cap = cv.VideoCapture(file_name)
    res = 0

    while True:
        ret,img = cap.read()

        if not ret:
            break

        res = res+1
                                            
    return res

file_name = "pedestrian.mp4"
tot_frame = total_frames(file_name)
print(f"Total Frames in {file_name} are {tot_frame}")