import dlib
import cv2
import numpy as np
import mysql.connector
from datetime import datetime

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",  # Replace with your MySQL password
    database="attendance_system"
)
cursor = db.cursor()

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor_path = 'Models/shape_predictor_68_face_landmarks.dat'  # Update this path
sp = dlib.shape_predictor(shape_predictor_path)

# Load Dlib's face recognition model
face_encoder = dlib.face_recognition_model_v1(
    'Models/dlib_face_recognition_resnet_model_v1.dat')  # Update this path


def mark_attendance():
    # Load registered employee face encodings from the database
    cursor.execute("SELECT employee_id, name, face_encoding FROM employees")
    employees = cursor.fetchall()

    known_face_encodings = []
    known_face_names = []
    employee_ids = []

    for employee in employees:
        employee_id = employee[0]
        name = employee[1]
        face_encoding = np.frombuffer(employee[2], dtype=np.float64)

        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
        employee_ids.append(employee_id)

    # Start webcam for face recognition
    video_capture = cv2.VideoCapture(0)
    print("Please look into the camera for attendance...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            print("No faces detected, please try again.")
            cv2.imshow('Attendance', frame)
            # Add a small delay to avoid overwhelming the system
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for face in faces:
            # Get face landmarks and encoding
            shape = sp(gray, face)
            face_encoding = np.array(face_encoder.compute_face_descriptor(frame, shape))

            # Compare with known face encodings
            matches = [np.linalg.norm(known_face_encoding - face_encoding) < 0.6 for known_face_encoding in
                       known_face_encodings]

            if any(matches):
                best_match_index = matches.index(True)
                employee_name = known_face_names[best_match_index]
                employee_id = employee_ids[best_match_index]

                # Get today's date
                today = datetime.now().date()

                # Check if attendance has already been marked for today
                cursor.execute("SELECT * FROM attendance WHERE employee_id = %s AND date = %s", (employee_id, today))
                attendance_record = cursor.fetchone()

                if attendance_record is None:  # If no record exists for today
                    # Mark attendance
                    cursor.execute(
                        "INSERT INTO attendance (employee_id, date, time) VALUES (%s, CURDATE(), CURTIME())",
                        (employee_id,)
                    )
                    db.commit()

                    print(f"Attendance marked for {employee_name} on {today} at {datetime.now().time()}.")
                    # Display a message on the frame
                    cv2.putText(frame, f'Attendance marked for {employee_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

                else:
                    print(f"Attendance for {employee_name} already marked today.")
                    cv2.putText(frame, f'Attendance already marked for {employee_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)

        # Display the frame with the face recognition result
        cv2.imshow('Attendance', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# Call the function to mark attendance
mark_attendance()
