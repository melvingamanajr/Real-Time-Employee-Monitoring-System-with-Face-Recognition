from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from .models import Employee
from django.utils import timezone
from django.shortcuts import redirect, get_object_or_404
import cv2
import base64
import time
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle
from sklearn.svm import SVC
from PIL import Image
import io

cam_location = 'Library'
path = "dataset"
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(os.path.join(path, cl))
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print('Images:', len(classNames))


def findEncodings(images):
    encodeList = []
    if os.path.exists('encodings_cache.pkl'):
        with open('encodings_cache.pkl', 'rb') as f:
            encodeList = pickle.load(f)
    else:
        for i, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode_start_time = time.time()
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            print(f'Cache encoding {i + 1}/{len(images)} took {time.time() - encode_start_time:.2f}s')
        with open('encodings_cache.pkl', 'wb') as f:
            pickle.dump(encodeList, f)
    return encodeList


encodeListKnown = findEncodings(images)
train_names = classNames

clf = SVC(kernel='linear', probability=True)
clf.fit(encodeListKnown, train_names)

print('Encoding Complete')


def home(request):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    return render(request, 'index.html', {'current_time': current_time, 'current_date': current_date})


def gen_frames(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    today = datetime.now().strftime("%Y-%m-%d")
    saved_today = False  # Flag to track if an employee has been saved today
    unknown = "Unknown"
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        height, width, _ = img.shape

        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            facesDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(facesDis)

            if facesDis[matchIndex] < 0.4:
                name = classNames[matchIndex]
                percentage = (1 - facesDis[matchIndex]) * 100
                identified = f"{name} ({percentage:.2f}%)"

                top, right, bottom, left = faceLoc
                center_x = int((left + right) / 2)
                center_y = int((top + bottom) / 2)
                box_width = int((right - left) / 2)
                box_height = int((bottom - top) / 2)

                if not saved_today:
                    # Crop the recognized face from the original image
                    face_img = img[top:bottom, left:right]

                    # Save image to a binary format
                    _, img_encoded = cv2.imencode('.jpeg', face_img)
                    image_blob = img_encoded.tobytes()

                    # Insert the data into the Employee model
                    Employee.objects.create(name=name, image=image_blob, location=cam_location)

                    saved_today = True  # Set the flag to indicate an employee has been saved today

                cv2.rectangle(img, (center_x - box_width, center_y - box_height),
                              (center_x + box_width, center_y + box_height), (0, 255, 0), 2)
                cv2.putText(img, identified, (center_x - box_width + 6, center_y - box_height - 6),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

            else:
                top, right, bottom, left = faceLoc
                center_x = int((left + right) / 2)
                center_y = int((top + bottom) / 2)
                box_width = int((right - left) / 2)
                box_height = int((bottom - top) / 2)
                cv2.rectangle(img, (center_x - box_width, center_y - box_height),
                              (center_x + box_width, center_y + box_height), (0, 0, 255), 2)
                cv2.putText(img, unknown, (center_x - box_width + 6, center_y - box_height - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                face_img = img[top:bottom, left:right]

                _, img_encoded = cv2.imencode('.jpeg', face_img)
                image_blob = img_encoded.tobytes()

                Employee.objects.create(name=unknown, image=image_blob, location=cam_location)

        # Convert the processed frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Yield the frame to the web application
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Reset the saved_today flag at the start of a new day
        if today != datetime.now().strftime("%Y-%m-%d"):
            saved_today = False
            today = datetime.now().strftime("%Y-%m-%d")


def delete_employee(request, employee_id):
    employee = get_object_or_404(Employee, id=employee_id)
    employee.delete()
    return redirect('log')


@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type="multipart/x-mixed-replace;boundary=frame")


def log(request):
    employee_data = Employee.objects.all()

    images = []
    for row in employee_data:
        image_data = row.image
        img = Image.open(io.BytesIO(image_data))
        images.append(img)

    return render(request, 'log.html', {'employee_data': employee_data, 'images': images})


def monitor(request):
    if request.method == 'POST':
        selected_date = request.POST['datetime']

        employee_data = Employee.objects.filter(datetime__date=selected_date)

        return render(request, 'monitor.html', {'employee_data': employee_data, 'selected_date': selected_date})

    return render(request, 'monitor.html')


def about(request):
    return render(request, 'about.html')


