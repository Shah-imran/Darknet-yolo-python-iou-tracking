import socket
import json
import base64
import datetime
import random
import pickle
import queue
import time
import pymysql

HOST = '116.73.57.17'  # The server's hostname or IP address
PORT = 5002         # The port used by the server
# imagePath = "./imageDir/2019-09-01/1/47/Entering1.jpg"
# imageList = "./imageDir/imageLinks.txt"
# mainDir = "./imageDir"

def json_message(imagePath):
    with open(imagePath, "rb") as image:
        b64string = base64.b64encode(image.read()).decode()

    name = { "RequestId": str(random.randint(10000,999999999)), 
        "MethodName":"OCRByPhoto" ,  
        "Parameters": { "PhotoByte64String": b64string }
        }

    json_data = json.dumps(name, sort_keys=False, indent=2)
    try:
        plateNumber = send_message(json_data + ";")
        return plateNumber
    except Exception as e:
        return ''

def send_message(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect((HOST, PORT))
        s.sendall(data.encode())
        temp = str()
        while data:
            data = s.recv(4096)
            if "PlateNumber" in data.decode():
                data = data.decode()
                break
        else:
            print("here")
    data = data.split(",")
    for i in data:
        if "PlateNumber" in i:
            data = i.replace('"PlateNumber": ', "").replace(" ", "").replace('"', "").replace("\n","")
            plateNumber = data
            # print("in ocr--", plateNumber)
    return plateNumber
def dbPush(plateNumber, imagePath, date, timestamp): 
    try:
        print(plateNumber, imagePath, type(date), type(timestamp))
        db = pymysql.connect('localhost', 'root', '', 'vehicle_backend')
        cursor = db.cursor()
        mySql_insert_query = """INSERT INTO vehiclelogs (plateNumber, Date, timestamp, imagePath) 
                                        VALUES (%s, %s, %s, %s) """
        recordTuple = (plateNumber, date, timestamp, imagePath)
        cursor.execute(mySql_insert_query, recordTuple)
        db.commit()
        print("Record inserted successfully into vehicleLogs table")
        return True
    except Exception as e:
        print(e)
        return False

# def readList():
#     while True:
#         with open(imageList, "r", encoding= "utf-8") as f:
#             data = f.read()
#         data = data.split("\n")
#         for i in range(len(data)):
#             if data[i] == "":
#                 data.pop(i)
#         print(data)
#         time.sleep(200)


if __name__ == "__main__":
    readList()           
