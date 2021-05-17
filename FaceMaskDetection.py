# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import playsound
import sqlite3
from datetime import datetime
import PySimpleGUI as sg
import tensorflow
import csv

parentPath = 'D:/FaceMaskDetection/'
folder = 'MaskRecord'
filePath = os.path.join(parentPath, folder)

if not os.path.exists(filePath):
    os.mkdir(filePath)

connection = sqlite3.connect(
    os.path.join(filePath,
                 'Mask' + str(datetime.now().strftime('%Y_%m_%d')) + '.db'))
cursor = connection.cursor()
time = time.time()
now = datetime.now()


def create_table():
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS maskDetectionRecord(datestamp TEXT, timestamp TEXT, status TEXT, totalMask INT, totalNoMask INT)'
    )


def data_entry(status, totalValueMask, totalValueNoMask):
    datestamp = str(now.strftime("%Y/%m/%d"))
    timestamp = str(now.strftime("%H:%M:%S"))
    status = status
    totalMask = totalValueMask
    totalNoMask = totalValueNoMask
    cursor.execute(
        "INSERT INTO maskDetectionRecord (datestamp, timestamp, status, totalMask, totalNoMask) VALUES (?, ?, ?, ?, ?)",
        (datestamp, timestamp, status, totalMask, totalNoMask))
    connection.commit()


def convertto_csv():

    # File path and name.
    fileName = os.path.join(
        filePath, ('Mask' + str(datetime.now().strftime('%Y_%m_%d')) + '.csv'))

    # Database.
    database = os.path.join(
        filePath, 'Mask' + str(datetime.now().strftime('%Y_%m_%d')) + '.db')
    connect = None

    # Check if the file path exists.
    if os.path.exists(filePath):

        try:

            # Connect to database.
            connect = sqlite3.connect(
                os.path.join(
                    filePath,
                    'Mask' + str(datetime.now().strftime('%Y_%m_%d')) + '.db'))

        except sqlite3.DatabaseError as e:

            # Confirm unsuccessful connection and quit.
            print("Database connection unsuccessful.")
            quit()

        # Cursor to execute query.
        cursor = connect.cursor()

        # SQL to select data from the person table.
        sqlSelect = \
            "SELECT datestamp, timestamp, status, totalMask, totalNoMask \
            FROM maskDetectionRecord \
            ORDER BY datestamp"

        try:

            # Execute query.
            cursor.execute(sqlSelect)

            # Fetch the data returned.
            results = cursor.fetchall()

            # Extract the table headers.
            headers = [i[0] for i in cursor.description]

            # Open CSV file for writing.
            csvFile = csv.writer(open(fileName, 'w', newline=''),
                                 delimiter=',',
                                 lineterminator='\r\n',
                                 quoting=csv.QUOTE_ALL,
                                 escapechar='\\')

            # Add the headers and data to the CSV file.
            csvFile.writerow(headers)
            csvFile.writerows(results)

            # Message stating export successful.
            # print("Data export successful.")

        except sqlite3.DatabaseError as e:

            # Message stating export unsuccessful.
            print("Data export unsuccessful.")
            quit()

        finally:

            # Close database connection.
            connect.close()

    else:

        # Message stating file path does not exist.
        print("File path does not exist.")


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (224, 224),
        (104.0, 177.0,
         123.0))  # preprocess data by using mean subtraction and scaling

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds, faces)


# load our serialized face detector model from disk
prototxtPath = "D:/FaceMaskDetection/face_detector/deploy.prototxt"
weightsPath = "D:/FaceMaskDetection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# create database
create_table()

# load the face mask detector model from disk
maskNet = load_model("D:/FaceMaskDetection/mask_detector.model")

# collect statistics
countMask = 0
countNoMask = 0
noMask = 0
tmpNoMask = 0

prev_faces = []
same_person = True
finish = False

# First the window layout in 2 columns
sg.theme('LightGrey')

video_column = [
    [sg.Image(filename='', key='image')],
]

result_column = [[
    sg.Text("Face Mask Detection",
            size=(10, 3),
            font='Helvetica 28 bold',
            justification='center'),
],
                 [
                     sg.Frame(layout=[[
                         sg.Text(key='withMask',
                                 font='Helvetica 20',
                                 text_color='Green',
                                 justification='right',
                                 size=(13, 1))
                     ]],
                              title='Total with mask')
                 ],
                 [
                     sg.Frame(layout=[[
                         sg.Text(key='withoutMask',
                                 font='Helvetica 20',
                                 text_color='Red',
                                 justification='right',
                                 size=(13, 1))
                     ]],
                              title='Total without mask')
                 ]]

# Full layout
layout = [[
    sg.Column(video_column),
    sg.VSeperator(),
    sg.Column(result_column),
]]

window = sg.Window('Face Mask Detection',
                   icon="D:/FaceMaskDetection/app.ico",
                   location=(400, 250)).Layout(layout)

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:  # The PSG "Event Loop"
    frame = vs.read()

    # get events for the window with 20ms max wait
    event, values = window.Read(timeout=20, timeout_key='timeout')

    if event is None:
        break  # if user closed window, quit

    (locs, preds, faces) = detect_and_predict_mask(frame, faceNet, maskNet)

    if len(prev_faces) == 0 and len(faces) > 0:
        same_person = False
    else:
        same_person = True

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        if mask > withoutMask:
            label = "Mask"
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0),
                          2)

            if same_person == False:
                countMask += 1
                finish = True

        else:
            label = "No Mask"
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255),
                          2)

            if same_person == False:
                tmpNoMask = countNoMask
                countNoMask += 1
                noMask += 1
                finish = True

                playsound.playsound("D:/FaceMaskDetection/alert.wav", False)

        if same_person == True and noMask > 0 and mask > withoutMask:
            countNoMask = tmpNoMask
            countMask += 1
            noMask = 0
            finish = True

        if finish == True:
            print("Mask: ", countMask)
            print("No mask: ", countNoMask)

            data_entry(label, countMask, countNoMask)
            window['withMask'].update(countMask)
            window['withoutMask'].update(countNoMask)
            finish = False

    prev_faces = faces

    window.FindElement('image').Update(data=cv2.imencode(
        '.png', frame)[1].tobytes())  # Update image in window

cursor.close()
connection.close()
convertto_csv()
window.close()
