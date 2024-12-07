import os
import json
import cv2
from flask import request
import face_recognition
import pyttsx3
import speech_recognition as sr

# File to store face database
DB_FILE = "face_database.json"

# Load database or create empty if not existing
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        face_db = json.load(f)
else:
    face_db = {}

def save_database():
    """Save the face database to a file."""
    with open(DB_FILE, "w") as f:
        json.dump(face_db, f, indent=4)

def add_face(name, encoding):
    """Add a new face to the database."""
    face_db[name] = encoding
    save_database()

def remove_face(name):
    """Remove a face from the database."""
    if name in face_db:
        del face_db[name]
        save_database()

def get_face_encoding(name):
    """Retrieve the encoding for a given face."""
    return face_db.get(name)

def list_faces():
    """List all faces in the database."""
    return list(face_db.keys())

def recognize_faces(frame):
    """Recognize faces in the given frame."""
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    recognized_faces = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(list(face_db.values()), face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = list(face_db.keys())[first_match_index]

        recognized_faces.append((name, face_location))

    return recognized_faces

def prompt_user_for_name():
    """Prompt the user to provide a name for the unknown face."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Please provide the name for the unknown person:")
        audio = recognizer.listen(source)

    try:
        name = recognizer.recognize_google(audio)
        return name
    except sr.UnknownValueError:
        print("Sorry, I could not understand the name.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None

def main():
    camera = request.get("http://{server_ip}:5000/")
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Set speaking rate (words per minute)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        recognized_faces = recognize_faces(frame)

        for name, (top, right, bottom, left) in recognized_faces:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if name == "Unknown":
                tts_engine.say("Unknown person detected. Please provide a name.")
                tts_engine.runAndWait()
                name = prompt_user_for_name()
                if name:
                    face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]
                    add_face(name, face_encoding.tolist())
                    tts_engine.say(f"Saved {name} to the database.")
                    tts_engine.runAndWait()
            else:
                distance = (right - left) / frame.shape[1]  # Estimate distance based on face width
                tts_engine.say(f"{name} is here, approximately {distance:.2f} meters away.")
                tts_engine.runAndWait()

        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()