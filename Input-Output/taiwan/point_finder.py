import cv2 as cv

def click_event(event,x,y,flags,params):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),4,(0,0,255),-1)
        print(f"{x} {y}")
        points.append([x,y])
        if len(points)<=4:
            cv.imshow('image',img)
        

points = []

img = cv.imread("static_frame_from_video.png")

cv.imshow('image',img)
cv.setMouseCallback('image',click_event)
cv.waitKey(0)
cv.destroyAllWindows()

# 1 3
# 1271 14
# 7 715
# 1276 714