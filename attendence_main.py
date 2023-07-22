
import datetime
import cv2
import face_recognition
import numpy as np
import csv

# initialize video capture
video_capture = cv2.VideoCapture(0)

# load known faces
harrys_image = face_recognition.load_image_file("faces/harry.jpeg")
harry_encoding = face_recognition.face_encodings(harrys_image)[0]
shreyas_image = face_recognition.load_image_file("faces/shreya.jpeg")
shreya_encoding = face_recognition.face_encodings(shreyas_image)[0]
Evas_image = face_recognition.load_image_file("faces/Eva.jpg")
Eva_encoding = face_recognition.face_encodings(Evas_image)[0]
Ayushs_image = face_recognition.load_image_file("faces/Ayush.jpg")
Ayush_encoding = face_recognition.face_encodings(Ayushs_image)[0]

known_face_encoding = [harry_encoding, shreya_encoding, Eva_encoding]
known_face_names = ["Harry", "Shreya", "Eva","Ayush"]

# list of expected students
students = known_face_names.copy()

# initialize attendance dictionary
attendance = {name: "Absent" for name in known_face_names}

face_locations = []
face_encodings = []

# get the current date and time
now = datetime.datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    # read a frame from the video capture
    ret, frame = video_capture.read()
    if not ret:
        print("Error: failed to capture frame")
        break

    # resize the frame and convert to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            attendance[name] = "Present"
            # students.remove(name)
            try:
                students.remove(name)
            except ValueError:
                pass

            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2

            # write attendance to CSV file
            lnwriter.writerow([name, current_date, now.strftime("%H:%M:%S"), attendance[name]])

    # display the frame with recognized face
    cv2.imshow("Attendance", frame)

    # wait for a key press and check if the 'q' key was pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # check if all students have been marked present
    if not students:
        break

# write attendance data for absent students to CSV file
for name in students:
    lnwriter.writerow([name, current_date, now.strftime("%H:%M:%S"), attendance[name]])

# close the CSV file and release the video capture
f.close()
video_capture.release()
cv2.destroyAllWindows()
