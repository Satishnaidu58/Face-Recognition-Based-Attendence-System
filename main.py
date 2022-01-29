import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
from datetime import date
from tkinter import *

from PIL import ImageTk

root = Tk()
root.geometry("733x566")
root.title("Face Detection")

Label(text="Face Detection and Recognition GUI\n", font="comicsansms 15 bold", pady=15, fg="Black").pack()


def real_face_detect():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    font = cv2.FONT_HERSHEY_SIMPLEX

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        count = 1
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, 'Face: ' + str(count), (x + w, y + h), font, 1, (255, 0, 0), 2)

            count += 1
        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# from PIL import ImageGrab
path = '/home/satish/python/project/face_recognition/Images/'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complete')


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        one = []

        for line in myDataList:
            dateList = []
            entry = line.split(',')
            dateList.append(entry[0])
            dateList.append(entry[2].replace('\n', ''))
            one.append(dateList)

        today = date.today()
        dateS = today.strftime("%b-%d-%Y")
        count = 0
        for data in one:
            if name == data[0]:
                if dateS != data[1]:
                    count += 1
                else:
                    count -= 1000
        if count >= 0:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString},{dateS}')
            return 1
        else:
            return 0


def real_face_recognition():
    # FOR CAPTURING SCREEN RATHER THAN WEBCAM
    # def captureScreen(bbox=(300,300,690+300,530+300)):
    #     capScr = np.array(ImageGrab.grab(bbox))
    #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    #     return capScr

    cap = cv2.VideoCapture(0)

    while True:

        success, img = cap.read()
        # img = captureScreen()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches and faceDis[matchIndex] < 0.50:
                name = classNames[matchIndex].upper()
                # markAttendance(name)
            else:
                name = 'Unknown'
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if name != 'Unknown':
                check = markAttendance(name)
                if check == 1:
                    print('Attendance Marked')

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break


def face():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread("Test/me.jpeg")

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches and faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
        else:
            name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        if name != 'Unknown':
            check = markAttendance(name)
            if check == 1:
                print('Attendance Marked')
    # resized_image = img.resize((300, 205), Image.ANTIALIAS)
    # new_image = ImageTk.PhotoImage(resized_image)
    cv2.imshow('Webcam', img)
    cv2.waitKey()


Button(text="Face Detection", command=real_face_detect).pack()
Label(text=" ").pack()
Button(text="Real time Face Recognition", command=real_face_recognition).pack()
Label(text=" ").pack()
Button(text="Face Recognition", command=face).pack()

root.mainloop()
