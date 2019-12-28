from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

from time import time, sleep
import argparse

from util import load_mot, iou

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
def cvDrawBoxes(detections, classCount, total_count, frame_num, classCount1,img):
    xTemp = 10
    yTemp = 10
    cv2.putText(img, "Count:"+str(total_count),(300, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [255, 0, 0], 2 )
    # for i in range(len(classCount)):
    #     cv2.putText(img, classes[i]+" : "+str(classCount[i]),(xTemp, yTemp), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                     [255, 0, 0], 2 )
    #     yTemp += 14

    # cv2.putText(img, "Frame Number:"+str(frame_num),(xTemp, yTemp), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                     [255, 0, 0], 2 )
    # cv2.putText(img, "ObjectsOnline:"+str(sum(classCount1)),(xTemp, yTemp+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                     [255, 0, 0], 2 )

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
        cv2.line(img,(right1,down1),(right2,down2),(0,0,255),5)
        # cv2.rectangle(img,(right1,down1),(right2,down2),(0,0,255),1)
        # print(detection[0].decode())
        # cv2.putText(img,
        #             detection[0].decode() +
        #             " [" + str(round(detection[1] * 100, 2)) + "] " + detection[3].decode(),
        #             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             [0, 255, 0], 2)
        # cv2.putText(img,
        #             detection[0].decode() + " " + detection[3].decode(),
        #             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             [0, 255, 0], 2)

        cv2.putText(img,
                    detection[0].decode(),
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
    global right1, down1, right2, down2, classes
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
    import datetime
    filename = str(datetime.datetime.now().date())
    filename = filename.replace(" ", "_")
    hourstamp = datetime.datetime.now().hour 
    with open(filename, "a", encoding="utf-8") as f:
        for i in classes:
            f.write(i+" ")
        f.write("\n")

    videoLink = "bike1.mp4"
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(videoLink)
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
            "4.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
            (darknet.network_width(netMain), darknet.network_height(netMain)))
    out1 = cv2.VideoWriter(
            "outputWithoutanno.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
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
    sigma_l, sigma_h, sigma_iou, t_min, memT = 0, 0.15, 0.1, 1, 100
    start = time()
    vehicleOnLine = list()
    pClassSecond = list()
    pClassSecondP = list()
    for i in range(len(classes)):
        pClassSecond.append(0)
        pClassSecondP.append(0)

    total_count = int()

    right1, down1, right2, down2 = 50, 150, 330, 150
    p1 = (right1, down1)
    q1 = (right2, down2)
    while True:
        prev_time = time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        # model gives detections results here
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.15)
        # print("-----detections------")
        # print(detections)
        # iou tracker implementation
        tempList = list()
        tempdict = dict()
        if str(datetime.datetime.now().date()) not in filename:
            filename = str(datetime.datetime.now().date())

        print(len(detections))
        for i in range(len(detections)):
            if detections[i][0].decode() != "person":
                x1, y1, x2, y2 = convertBack(detections[i][2][0],detections[i][2][1],detections[i][2][2],detections[i][2][3])
                # print(x1, y1, x2, y2, detections[i][0])
                # print(detections[i][2])
                tempdict["score"] = detections[i][1]
                tempdict["bbox"] = (float(x1), float(y1), float(x2), float(y2))
                tempdict["class"] = detections[i][0]
                # print(tempdict)
                tempList.append(tempdict.copy())
            # print(tempList)
        # print("detections------------")
        # print(len(tempList))
        # print(tempList)

        detections_frame = tempList
        dets = [det for det in detections_frame if det['score'] >= sigma_l]
        updated_tracks = []
        tracks_active = []
        tracks_finished = []
        print("frame num--------", frame_num)  
        print("dets")
        print(dets)
        print("tracks active1")
        print(tracks_active1)
        for track in tracks_active1:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
                # print("Track", track)
                # print(best_match['bbox'])
                # print("Iou----",iou(track['bboxes'][-1], best_match['bbox']))
                if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'], best_match['score'])

                    updated_tracks.append(track)
                    print("updated tracks---")
                    print(updated_tracks)
                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

            # # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min and frame_num - (len(track['bboxes'])+track['start_frame']) <= memT:
                    tracks_finished.append(track)
            # print("Track", track)
        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num, 'id': 0, "class": det['class']} for det in dets]
        print("new tracks--")
        print(new_tracks)
        tracks_active1 = updated_tracks + new_tracks
        # print("tracks_active - ", len(tracks_active1))
        # for i in tracks_active:
        #     print(i)
        # finish all remaining active tracks
        tracks_active += [track for track in tracks_active1
                            if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]
        print("tracks active 1")
        print(tracks_active1)
        print("tracks active")
        print(tracks_active)
        print("Inside tracks finished")
        print(tracks_finished)
        
        for track in tracks_finished:
            tracks_active1.append(track)

        # print("Tracks before ------")
        # print(len(tracks_finished))
        # for i in tracks_finished:
        #   print(i)
        # for i in tracks_finished:
        #     for j in range(tracks_finished.count(i)-1):
        #         tracks_finished.remove(i)
        # print("Tracks ------")
        # print(len(tracks_finished))
        # for i in tracks_finished:
        #   print(i)
        # print(tracks_active)
        
        
        track_Results = list()
        for trackT in tracks_active:
            # print(trackT)
            bbox = trackT["bboxes"][len(trackT['bboxes'])-1]
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
                           'wx': -1,
                           'wy': -1,
                           'wz': -1}

            # print(row['class'].decode())
            

            if time() - start >= 1:
                #saving the data with timsatmp
                if frame_num % 500 == 0:
                    vehicleOnLine = vehicleOnLine[len(vehicleOnLine)//2:len(vehicleOnLine)]
                for i in range(len(pClassSecond)):
                    pClassSecond[i] = 0
                xmin, ymin, xmax, ymax = convertBack(float(row['x']), float(row['y']), float(row['w']), float(row['h']))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymin)
                pt3 = (xmin, ymax)
                pt4 = (xmax, ymax)
                if (do_intersect(p1, q1, pt1, pt3) == True and do_intersect(p1, q1, pt2, pt4) == True ) and row['id'] not in vehicleOnLine:
                    temp = classes.index(row['class'].decode())
                    pClassSecond[temp]=pClassSecond[temp]+1 
                    pClassSecondP[temp]=pClassSecondP[temp]+1 
                    vehicleOnLine.append(row['id'])
                    total_count+=1 
                start = time()

            else:

                xmin, ymin, xmax, ymax = convertBack(float(row['x']), float(row['y']), float(row['w']), float(row['h']))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymin)
                pt3 = (xmax, ymin)
                pt4 = (xmax, ymax)
                if (do_intersect(p1, q1, pt1, pt3) == True and do_intersect(p1, q1, pt2, pt4) == True ) and row['id'] not in vehicleOnLine:
                    temp = classes.index(row['class'].decode())
                    pClassSecond[temp]=pClassSecond[temp]+1 
                    pClassSecondP[temp]=pClassSecondP[temp]+1 
                    vehicleOnLine.append(row['id'])
                    total_count+=1



            track_Results.append(row.copy())


        
        # print(pClassSecond)

        # for trackT in tracks_finished:
        #     for i, bbox in enumerate(trackT['bboxes']):
        #         row = {'id': id_,
        #                'frame': trackT['start_frame'] + 1,
        #                'x': bbox[0],
        #                'y': bbox[1],
        #                'w': bbox[2] - bbox[0],
        #                'h': bbox[3] - bbox[1],
        #                'score': trackT['max_score'],
        #                'wx': -1,
        #                'wy': -1,
        #                'wz': -1}
        #         # print("frame num--------", frame_num)
        #         track_Results.append(row.copy())
        #     id_ += 1
        # sleep(0.2)
        with open(filename, "a", encoding="utf-8") as f:
            for i in pClassSecond:
                f.write(str(i)+",")
            f.write(str(datetime.datetime.now()))
            f.write("," + str(frame_num))
            f.write("\n")
        
        # print(len(track_Results))
        track_resultD = list()
        for i in track_Results:
            # print(i) 
            track_resultD.append([bytes(str(i["id"]).encode('utf-8')), i['score'], (i["x"],i['y'],i['w'],i['h']), i['class']])
        # print(track_resultD)
        frame_num+=1
        out1.write(frame_resized)
        print(1/(time()-prev_time))
        image = cvDrawBoxes(track_resultD, pClassSecondP, total_count, frame_num, pClassSecond, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if frame_num == 150:
        #     break
        cv2.imshow('Demo', image)
        out.write(image)
        cv2.waitKey(3)

    # cv2.destroyAllWindows()
    cap.release()
    out.release()
    out1.release()

if __name__ == "__main__":
    YOLO()
