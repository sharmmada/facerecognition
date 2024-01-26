import cv2
import face_recognition
import os

known_face_encodings = []
known_face_names = []

def load_known_faces():
    voter_image_dir = r'C:\Users\user\PycharmProjects\face_recognition\facesss'
    for image_filename in os.listdir(voter_image_dir):
        voter_name = os.path.splitext(image_filename)[0]
        image_path = os.path.join(voter_image_dir, image_filename)
        voter_image = face_recognition.load_image_file(image_path)
        voter_face_encoding = face_recognition.face_encodings(voter_image)[0]  # Assuming one face per image
        known_face_encodings.append(voter_face_encoding)
        known_face_names.append(voter_name)
        print(known_face_names)


def face_check():
    load_known_faces()
    print("System is going to check face authentication")
    print("Please look at the camera")

    # Initialize system camera (use 0 for default camera)
    camera = cv2.VideoCapture(0)

    while True:
        try:
            ret, frame = camera.read()

            # Find all face locations and face encodings in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare face_encoding with known_face_encodings
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

                name = "Unknown"

                # Check if there's a match
                if True in matches:
                    match_index = matches.index(True)
                    name = known_face_names[match_index]

                    print(f"Authorized : {name}")  # Print the authorized voter's name in the shell

                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_check()
