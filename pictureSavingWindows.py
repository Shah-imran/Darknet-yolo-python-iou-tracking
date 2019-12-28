from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
# import darknet
import threading
import datetime
import queue
import ocr
from time import time, sleep
import argparse

from util import load_mot, iou

q = queue.Queue()

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


hasGPU = True

lib = CDLL("yolo_cpp_dll.dll", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = \
    [c_void_p, c_int, c_int, c_float, c_float, POINTER(
        c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    im, arr = array_to_image(image)
    if debug:
        print("Loaded image")
    num = c_int(0)
    if debug:
        print("Assigned num")
    pnum = pointer(num)
    if debug:
        print("Assigned pnum")
    predict_image(net, im)
    if debug:
        print("did prediction")
    # dets = get_network_boxes(
    #     net, image.shape[1], image.shape[0],
    #     thresh, hier_thresh,
    #     None, 0, pnum, 0)  # OpenCV
    dets = get_network_boxes(net, im.w, im.h,
                             thresh, hier_thresh, None, 0, pnum, 0)
    if debug:
        print("Got dets")
    num = pnum[0]
    if debug:
        print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug:
        print("did sort")
    res = []
    if debug:
        print("about to range")
    for j in range(num):
        if debug:
            print("Ranging on "+str(j)+" of "+str(num))
        if debug:
            print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug:
                print("Class-ranging on "+str(i)+" of " +
                      str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug:
        print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug:
        print("did sort")
    # free_image(im)
    if debug:
        print("freed image")
    free_detections(dets, num)
    if debug:
        print("freed detections")
    return res

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
            mainDir = "c:/xampp/htdocs/VehicleBackend/imageDir/"
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

if not os.path.exists("c:/xampp/htdocs/VehicleBackend/imageDir/"):
    try:
        os.mkdir("c:/xampp/htdocs/VehicleBackend/imageDir/")
        print("created")
    except Exception as e:
        print(e)



def YOLO():
    global right1, down1, right2, down2, classes, right3, down3, right4, down4, q

    global metaMain, netMain, altNames
    # configPath = "./cfg/yolov3.cfg"
    # weightPath = "./yolov3.weights"
    # metaPath = "./cfg/coco.data"
    configPath = "data/yolov3-tiny_customclass.cfg"
    weightPath = "data/yolov3-tiny_customclass_65800.weights"
    metaPath = "data/objJust9.data"
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
        netMain = load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
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

    right1, down1, right2, down2 = 50, 150, 330, 150
    p1 = (right1, down1)
    q1 = (right2, down2)

    right3, down3, right4, down4 = 50, 200, 330, 200
    p2 = (right3, down3)
    q2 = (right4, down4)

    threading.Thread(target=saveImage, daemon=True).start()

    while True:
        prev_time = time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (lib.network_width(netMain),
                                    lib.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        # darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        # model gives detections results here
        detections = detect(netMain, metaMain, darknet_image, thresh=0.15)
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
                if (row['y'] - 200 <= 0 and row['y'] - 200 >= -25 ) and row['id'] not in vehicleOnLine:
                    # print(row['y'] - 180)
                    vehicleOnLine.append(row['id'])
                    total_count+=1 
                    print("matched")
                    q.put([imageDirectory, frame_read, row['direction']+str(row['id'])+".jpg", timestamp])
                    # threading.Thread(target=saveImage, args=[imageDirectory, frame_read, str(row['id'])+".jpg", ], daemon=True).start()
            elif row['direction'] == "Getting_Out":
                if (row['y'] - 150 <= 25 and row['y'] - 150 >= 0 ) and row['id'] not in vehicleOnLine:
                    # print(row['y'] - 150)
                    vehicleOnLine.append(row['id'])
                    total_count+=1 
                    print("matched")
                    q.put([imageDirectory, frame_read, row['direction']+str(row['id'])+".jpg", timestamp])
                    # threading.Thread(target=saveImage, args=[imageDirectory, frame_read, str(row['id'])+".jpg", ], daemon=True).start()
            else:
                if (row['y'] - 200 <= 15 and row['y'] - 200 >= -15 ) and row['id'] not in vehicleOnLine:
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
