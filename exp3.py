# import the necessary modules
import freenect
import cv2
import numpy as np
import math
import vlc
import time
import webbrowser
import playsound


# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


# function to get depth image from kinect
def get_depth(value):
    depth, _ = freenect.sync_get_depth()
    rgbframes, _ = freenect.sync_get_video()
    rgbframes = cv2.cvtColor(rgbframes, cv2.COLOR_RGB2BGR)
    #print(depth)
    # print (depth.shape) #(480,640)
    # np.clip(array, 0, 2**10 - 1, array)
    # array >>= 2
    # center_pixel_depth = depth[width/2, height/2]
    # depth_mask = np.where(depth == center_pixel_depth, 255,     0).astype(np.uint8)
    depth_mask = np.where(depth < 650, 255, 0).astype(np.uint8)
    # ret,thresh1 = cv2.threshold(depth_mask,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #tp,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #drawing = depth_mask.copy()
    thresh1=depth_mask.copy()
    # _, thresh1 = cv2.threshold(depth_mask, 127, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    image, contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(cnt)
    rec_tuple=(x,y,x+w,y+h)
    #print (rec_tuple)
    #print (cnt)
    cv2.rectangle(thresh1, (x, y), (x + w, y + h), (255, 255, 255), 0)
    cv2.rectangle(rgbframes, (x, y), (x + w, y + h), (255, 0, 0), 0)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(depth_mask.shape, np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
    cv2.drawContours(drawing,[hull],0,(0,0,255),0)
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)
    cv2.drawContours(rgbframes, contours, -1, (0, 255, 0), 3)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(thresh1, far, 1, [127, 127, 127], -1)
            cv2.circle(rgbframes, far, 1, [255, 0, 0], -1)

        # dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(thresh1, start, end, [255, 255, 255], 2)
        cv2.line(rgbframes, start, end, [0, 0, 255], 2)
        # cv2.circle(crop_img,far,5,[0,0,255],-1)
    cv2.imshow('Thresholded', thresh1)
    if value == 0:
        cv2.putText(rgbframes, "Hiiiii!!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0), 5)
    elif value == 1:
        cv2.putText(rgbframes,"How are You?", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),5)
    elif value == 2:
        cv2.putText(rgbframes, "I am fine, thankyou!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),4)
    elif value==3:
        cv2.putText(rgbframes, "nonennenenene", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    cv2.imshow('rgbframes', rgbframes)
    if count_defects == 1:
        return 0
    elif count_defects == 3:
        return 1
    elif count_defects == 5:
        return 2
    else:
        return 3

if __name__ == "__main__":
    matrix=[0,0,0,0]
    value=2
    i=0
    while(1):
        print ("i=",i," value=",value)
        if(i%30==0):
            j = 1
            maxind=0
            print ("matrix is ",matrix)
            while(j<3):
                if(matrix[j] > matrix[maxind]):
                    maxind=j
                j+=1
            value=maxind
            if value==0:
               # playsound.playsound("hi.mp3")
               pass
            if value==1:
                p = vlc.MediaPlayer("/home/digvijay/Documents/howAreYou.mp3")
                p.play()
            if value==2:
                p = vlc.MediaPlayer("/home/digvijay/Documents/iAmFine.mp3")
                p.play()

            matrix=[0,0,0,0]
        else:
            index=get_depth(value)
            print ("index=",index)
            matrix[index]+=1
        i+=1
        # get a frame from RGB camera
        #frame = get_video()
        # get a frame from depth sensor
        #depth = get_depth()
        # display RGB image
        #cv2.imshow('RGB image', frame)
        # display depth image
        #cv2.imshow('Depth image', depth)
        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
            cv2.destroyAllWindows()