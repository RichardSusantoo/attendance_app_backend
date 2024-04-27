from flask import Flask, request, jsonify, json
from flask_cors import CORS, cross_origin
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import mysql.connector
from mysql.connector import Error
import pandas as pd
import joblib
import bcrypt
from flask import Response
from datetime import date
from datetime import datetime, timedelta
import time

app = Flask(__name__)

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


nimgs = 10

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}}, CORS_SUPPORTS_CREDENTIALS=True)
app.config['CORS_HEADERS'] = 'Content-Type'
try:
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='attendance_data'
    )
    print("Connection to MySQL DB successful")
except Error as e:
    print(f"The error '{e}' occurred")


if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


@app.route("/registeruser", methods=["POST","GET"])
@cross_origin()
def submitData():
    response_object = {'status':'success'}
    if request.method == "POST":
        post_data = request.get_json()
        fullname   = post_data.get('fullname')
        nim  = post_data.get('nim')
        email = post_data.get('email')
        password = post_data.get('password')
        
        userimagefolder = 'static/faces/'+fullname+'_'+str(nim)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Adding new User', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Adding new User', cv2.WND_PROP_TOPMOST, 1.0)
        i, j = 0, 0
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = fullname+'_'+str(i)+'.jpg'
                    face_image = frame[y:y+h, x:x+w]
                    face_data = cv2.imencode('.jpg', face_image)[1].tostring()
                    with open(userimagefolder+'/'+name, 'wb') as f:
                        f.write(face_data)
                    i += 1
                j += 1
            if j == nimgs*5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        train_model()
        
        cursor = connection.cursor()
        sql_query = "INSERT INTO users (fullname, nim, email, password, face_data) VALUES (%s, %s, %s, %s, %s)"
        val = (fullname, nim, email, hashed_password, face_data)
        cursor.execute(sql_query, val)
        connection.commit()
        response_object['message'] ='Data added!'
        cursor.close()
    return jsonify(response_object)

@app.route("/login", methods=["POST"])
@cross_origin()
def login():
    response_object = {'status': 'fail', 'message': 'Invalid credentials'}
    if request.method == "POST":
        post_data = request.get_json()
        email = post_data.get('email')
        password = post_data.get('password')
        
        cursor = connection.cursor()
        sql_query = "SELECT * FROM users WHERE email = %s"
        val = (email,)
        cursor.execute(sql_query, val)
        user = cursor.fetchone()
        
        if user:
            hashed_password = user[4].encode('utf-8')
            print(hashed_password)
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                response_object['status'] = 'success'
                response_object['message'] = 'Login successful'
                response_object['user'] = {'user_id': user[0], 'fullname': user[1], 'nim': user[2], 'email': user[3]}
            else:
                response_object['message'] = 'Invalid password'
        else:
            response_object['message'] = 'User not found'
        cursor.close()
    
    return jsonify(response_object)

@app.route("/jadwalkelas", methods=["GET"])
@cross_origin()
def getclassschedule():
    response_object = {'status':'success', 'data': []}
    if request.method == "GET":
        nim = request.args.get('nim')
        if nim:
            cursor = connection.cursor()
            sql_query = "SELECT class_name, class_time FROM jadwalkelas WHERE nim = %s"
            val = (nim,)
            cursor.execute(sql_query, val)
            class_schedule = cursor.fetchall()
            cursor.close()

            for class_ in class_schedule:
                class_time_str = str(class_[1])
                response_object['data'].append({
                    'class_name': class_[0],
                    'class_time': class_time_str
                })
        else:
            response_object['status'] = 'fail'
            response_object['message'] = 'nim is required'

    return jsonify(response_object)

@app.route("/users", methods=["GET"])
@cross_origin()
def get_users():
    response_object = {'status': 'success', 'data': []}
    if request.method == "GET":
        nim = request.args.get('nim')
        if nim:
            cursor = connection.cursor()
            sql_query = "SELECT fullname, nim, email FROM users WHERE nim = %s"
            val = (nim,)
            cursor.execute(sql_query, val)
            users = cursor.fetchall()
            cursor.close()

            for user in users:
                response_object['data'].append({
                    'fullname': user[0],
                    'nim': user[1],
                    'email': user[2]
                })
        else:
            response_object['status'] = 'fail'
            response_object['message'] = 'nim is required'

    return jsonify(response_object)

@app.route('/mark-attendance', methods=['GET'])
def mark_attendance():
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Attendance')
    cv2.setWindowProperty('Attendance', cv2.WND_PROP_TOPMOST, 1)
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            identified_person_name = identified_person.split('_')[0]
            identified_person_nim = identified_person.split('_')[1]
            add_attendance(identified_person_name, identified_person_nim)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        cv2.imshow('Attendance', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or time.time() - start_time > 2:  # 27 is the ASCII value for the escape key
            break
    cap.release()
    cv2.destroyAllWindows()
    return 'Attendance marked!'

def add_attendance(name, nim):
    cursor = connection.cursor()

    query = "SELECT * FROM users WHERE fullname=%s AND nim=%s"
    cursor.execute(query, (name, nim))
    result = cursor.fetchone()

    if result:
        utc_time = datetime.utcnow()
        utc_plus_seven = utc_time + timedelta(hours=7)
        today = utc_plus_seven.date()
        five_minutes_ago = utc_plus_seven - timedelta(minutes=5)
        query = "SELECT * FROM attendancehistory WHERE fullname=%s AND nim=%s AND AttendanceDate>=%s AND AttendanceDate<%s"
        cursor.execute(query, (name, nim, five_minutes_ago, today + timedelta(days=1)))
        attendance_today = cursor.fetchone()

        if not attendance_today:
            query = "INSERT INTO attendancehistory (fullname, nim, AttendanceDate) VALUES (%s, %s, %s)"
            cursor.execute(query, (name, nim, utc_plus_seven))
            print('Attendance marked successfully!')
            connection.commit()
        else:
            print(f"{name} already marked attendance within the last 5 minutes.")
    else:
        print(f"{name} not found in users table")

    cursor.close()

if __name__ == "__main__":
    app.run(debug=True)