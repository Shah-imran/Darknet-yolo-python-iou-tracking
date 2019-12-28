from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import threading
import datetime
import queue
import ocr
from time import time, sleep
import argparse

from util import load_mot, iou

q = queue.Queue()

def on_segment(p, q, r):
    '''Given three colinear points p, q, r, the function checks if 
    point q lies on line segment "pr"
    '''
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    '''Find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
    '''

    val = ((q[1] - p[1]) * (r[0] - q[0]) - 
            (q[0] - p[0]) * (r[1] - q[1]))
    if val == 0:
        return 0  # colinear
    elif val > 0:
        return 1   # clockwise
    else:
        return 2  # counter-clockwise

def do_intersect(p1, q1, p2, q2):
    '''Main function to check whether the closed line segments p1 - q1 and p2 
       - q2 intersect'''
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2 and o3 != o4):
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 and on_segment(p1, p2, q1)):
        return True

    # p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 and on_segment(p1, q2, q1)):
        return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 and on_segment(p2, p1, q2)):
        return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 and on_segment(p2, q1, q2)):
        return True

    return False # Doesn't fall in any of the above cases

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

# opencv 
def cvDrawBoxes(detections, total_count, img):
    xTemp = 10
    yTemp = 10
    cv2.putText(img, "Count:"+str(total_count),(300, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [255, 0, 0], 2 )

    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))

        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        height, width, channels = img.shape
        p1 = (right1, down1)
        q1 = (right2, down2)
        # print(xmin, ymin, xmax, ymax)

        cv2.rectangle(img, pt1, pt2, (0, 255, 255), 1)
        cv2.line(img,(right1,down1),(right2,down2),(255,0,0),5)
        cv2.line(img,(right3,down3),(right4,down4),(0,255,0),5)

        cv2.putText(img,
                    detection[0].decode() + " " + detection[4],
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        
    return img

#converting coordinates
def convertToCenter(bbox):
    x = (bbox[0]+bbox[2])/2
    y = (bbox[1]+bbox[3])/2
    return x, y

def saveImage():
    global q
    while True:
        while not q.empty():
            data = q.get()
            # print(data)
            imageDirectory, frame_read, filename, timestamp = data[0], data[1], data[2], data[3]
            date, hour, minute = imageDirectory.split("/")
            # print(imageDirectory, filename)
            mainDir = "/opt/lampp/htdocs/VehicleBackend/imageDir/"
            if not os.path.exists(mainDir+date):
                while True:
                    try:
                        os.mkdir(mainDir+date)
                        break
                    except:
                        pass
            if not os.path.exists(mainDir+date+"/"+hour):
                while True:
                    try:
                        os.mkdir(mainDir+date+"/"+hour)
                        break
                    except:
                        pass
            if not os.path.exists(mainDir+date+"/"+hour+"/"+minute):
                while True:
                    try:
                        os.mkdir(mainDir+date+"/"+hour+"/"+minute)
                        break
                    except:
                        pass
            while True:
                try:
                    imagePath = mainDir + imageDirectory+"/"+filename
                    cv2.imwrite(imagePath, frame_read) 
                    break
                except:
                    pass
            
            try:
                plateNumber = ocr.json_message(imagePath)
                print("Plate Number - ", plateNumber)
                # print(imagePath)
                # imagePath = os.path.join(os.getcwd(),imagePath[2:])
                status = ocr.dbPush(plateNumber, "/VehicleBackend/imageDir/"+ imageDirectory+"/"+filename, date, timestamp)
                if status == False:
                    with open("./imageDir/imageLinks.txt", "a", encoding='utf-8') as f:
                        f.write(imagePath +" "+plateNumber+"\n")
            except Exception as e:
                print(e, "---hkjfds")

        sleep(1)
        # cv2.imwrite(imageDirectory+filename, frame_read)
    pass

netMain = None
metaMain = None
altNames = None

if not os.path.exists("/opt/lampp/htdocs/VehicleBackend/imageDir/"):
    try:
        os.mkdir("/opt/lampp/htdocs/VehicleBackend/imageDir/")
        print("created")
    except Exception as e:
        print(e)

def YOLO():
    global right1, down1, right2, down2, classes, right3, down3, right4, down4, q
    global metaMain, netMain, altNames
    # configPath = "./cfg/yolov3.cfg"
    # weightPath = "./yolov3.weights"
    # metaPath = "./cfg/coco.data"
    # model code
    configPath = "./yolov3-custom_tiny.cfg"
    weightPath = "./yolov3-tiny_customclass_65800.weights"
    metaPath = "./customModel.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                            global classes
                            classes = altNames
                except TypeError:
                    pass
        except Exception:
            pass
    # model end
    

    videoLink = "1.mp4"
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(videoLink)
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
            "4.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
            (darknet.network_width(netMain), darknet.network_height(netMain)))

    print("Starting the YOLO loop...")
    print(darknet.network_width(netMain),darknet.network_height(netMain))

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    tracks_active1 = []
    tracks_finished = []
    frame_num = 1
    id_ = 1
    sigma_l, sigma_h, sigma_iou, t_min = 0.0, 0.0, 0.1, 3
    start = time()
    vehicleOnLine = list()
    total_count = int()
    #red line
    right1, down1, right2, down2 = 10, 150, 400, 150
    p1 = (right1, down1)
    q1 = (right2, down2)
    #green one
    right3, down3, right4, down4 = 10, 200, 400, 200
    p2 = (right3, down3)
    q2 = (right4, down4)

    threading.Thread(target=saveImage, daemon=True).start()

    while True:
        prev_time = time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,(darknet.network_width(netMain),
                            darknet.network_height(netMain)),interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        # model gives detections results here
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.15)
        # iou tracker implementation
        tempList = list()
        tempdict = dict()

        # print(len(detections))
        for i in range(len(detections)):
            if detections[i][0].decode() != "person":
                x1, y1, x2, y2 = convertBack(detections[i][2][0],detections[i][2][1],detections[i][2][2],detections[i][2][3])
                tempdict["score"] = detections[i][1]
                tempdict["bbox"] = (float(x1), float(y1), float(x2), float(y2))
                tempdict["class"] = detections[i][0]
                tempList.append(tempdict.copy())

        detections_frame = tempList
        dets = [det for det in detections_frame if det['score'] >= sigma_l]
        updated_tracks = []
        tracks_active = []
        for track in tracks_active1:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
                if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'], best_match['score'])

                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num, 'id': 0, "class": det['class']} for det in dets]
        tracks_active1 = updated_tracks + new_tracks
        # print("tracks_active - ", len(tracks_active1))
        tracks_active += [track for track in tracks_active1
                            if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]


        
        
        
        track_Results = list()
        for trackT in tracks_active:
            # print("trackT")
            # print(trackT["bboxes"])
            direction = 'Unknown'
            bbox = trackT["bboxes"][len(trackT['bboxes'])-1]
            x, y = convertToCenter(bbox)
            x1, y1 = convertToCenter(trackT['bboxes'][0])
            if y < y1:
                direction = "Getting_Out"
            elif y > y1:
                direction = "Entering"
            else:
                direction = "Unknown"
            
            index = tracks_active.index(trackT)
            temp = tracks_active[index]
            if temp['id'] == 0:
                tracks_active[index]["id"] = id_
                id_+=1

                row = {'id': tracks_active[index]["id"],
                           'frame': frame_num,
                           'x': (bbox[0]+bbox[2])/2,
                           'y': (bbox[1]+bbox[3])/2,
                           'w': bbox[2] - bbox[0],
                           'h': bbox[3] - bbox[1],
                           'score': trackT['max_score'],
                           'class': trackT['class'],
                           'direction': direction,
                           'wx': -1,
                           'wy': -1,
                           'wz': -1}

            else:
                row = {'id': temp["id"],
                           'frame': frame_num,
                           'x': (bbox[0]+bbox[2])/2,
                           'y': (bbox[1]+bbox[3])/2,
                           'w': bbox[2] - bbox[0],
                           'h': bbox[3] - bbox[1],
                           'score': trackT['max_score'],
                           'class': trackT['class'],
                           'direction': direction,
                           'wx': -1,
                           'wy': -1,
                           'wz': -1}

            xmin, ymin, xmax, ymax = convertBack(float(row['x']), float(row['y']), float(row['w']), float(row['h']))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymin)
            pt3 = (xmin, ymax)
            pt4 = (xmax, ymax)

            if time() - start >= 1:
                if frame_num % 500 == 0:
                    vehicleOnLine = vehicleOnLine[len(vehicleOnLine)//2:len(vehicleOnLine)]
                start = time()
            else: 
                pass
            
            
            date = str(datetime.datetime.now().date())
            hour = str(datetime.datetime.now().hour)
            minute = str(datetime.datetime.now().minute)
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            imageDirectory = date+"/"+hour+"/"+minute

            if row['direction'] == "Entering":
                if (do_intersect(p1, q1, pt1, pt3) == True and do_intersect(p1, q1, pt2, pt4) == True ) and row['id'] not in vehicleOnLine:
                    # print(row['y'] - 180)
                    vehicleOnLine.append(row['id'])
                    total_count+=1 
                    print("matched")
                    q.put([imageDirectory, frame_read, row['direction']+str(row['id'])+".jpg", timestamp])
                    # threading.Thread(target=saveImage, args=[imageDirectory, frame_read, str(row['id'])+".jpg", ], daemon=True).start()
            elif row['direction'] == "Getting_Out":
                if (do_intersect(p1, q1, pt1, pt3) == True and do_intersect(p1, q1, pt2, pt4) == True ) and row['id'] not in vehicleOnLine:
                    # print(row['y'] - 150)
                    vehicleOnLine.append(row['id'])
                    total_count+=1 
                    print("matched")
                    q.put([imageDirectory, frame_read, row['direction']+str(row['id'])+".jpg", timestamp])
                    # threading.Thread(target=saveImage, args=[imageDirectory, frame_read, str(row['id'])+".jpg", ], daemon=True).start()
            else:
                if (do_intersect(p1, q1, pt1, pt3) == True and do_intersect(p1, q1, pt2, pt4) == True ) and row['id'] not in vehicleOnLine:
                    # print(row['y'] - 180)
                    vehicleOnLine.append(row['id'])
                    total_count+=1 
                    print("matched")
                    q.put([imageDirectory, frame_read, row['direction']+str(row['id'])+".jpg", timestamp])
                    # threading.Thread(target=saveImage, args=[imageDirectory, frame_read, str(row['id'])+".jpg", ], daemon=True).start()

            track_Results.append(row.copy())

        # print("frame num--------", frame_num)  
        track_resultD = list()
        for i in track_Results: 
            track_resultD.append([bytes(str(i["id"]).encode('utf-8')), i['score'], (i["x"],i['y'],i['w'],i['h']), i['class'], i['direction']])
        
        frame_num+=1
        # print(1/(time()-prev_time))
        image = cvDrawBoxes(track_resultD, total_count, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if frame_num == 150:
        #     break
        cv2.imshow('Demo', image)
        out.write(image)
        cv2.waitKey(3)

    # cv2.destroyAllWindows()
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
