import dlib
import cv2
import numpy as np
import mysql.connector
import os

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",  # replace with your MySQL password
    database="attendance_system"
)
cursor = db.cursor()

# Load Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Update the path to where you saved the shape_predictor_68_face_landmarks.dat file
shape_predictor_path = 'Models/shape_predictor_68_face_landmarks.dat'  # Update this path
sp = dlib.shape_predictor(shape_predictor_path)

# Load Dlib's face encoder
face_encoder = dlib.face_recognition_model_v1(
    'Models/dlib_face_recognition_resnet_model_v1.dat')  # Update this path


# Function to register employee with face encoding
def register_employee():
    # Accepting employee name and ID as input during runtime
    employee_name = input("Enter employee name: ")
    employee_id = input("Enter employee ID: ")

    video_capture = cv2.VideoCapture(0)
    print("Please look into the camera to register...")

    ret, frame = video_capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            print("No face detected. Please try again.")
            return

        # Assuming the first detected face is the one we want
        for face in faces:
            shape = sp(gray, face)
            face_encoding = np.array(face_encoder.compute_face_descriptor(frame, shape))

            # Store the employee's ID, name, and face encoding in the database
            cursor.execute("INSERT INTO employees (employee_id, name, face_encoding) VALUES (%s, %s, %s)",
                           (employee_id, employee_name, face_encoding.tobytes()))
            db.commit()

            print(f"{employee_name} (ID: {employee_id}) has been registered successfully!")
            # Display the frame
            cv2.imshow('Registration', frame)

            # Wait indefinitely for a key press
            cv2.waitKey(0)

    video_capture.release()
    cv2.destroyAllWindows()



# Call the function to register an employee
register_employee()
