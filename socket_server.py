import socket
import cv2
import pickle
import struct
import threading
import time
import speech_recognition as sr
import pyttsx3
from servo_control import ServoPositionHandler

# Configuration settings
SERVER_IP = 'YOUR_SERVER_IP'
PORT_COMMAND = 5000
PORT_VIDEO = 5001

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Set speaking rate (words per minute)

# Servo setup()  # Assuming ServoController is defined elsewhere
servo = ServoPositionHandler()
def speak_text(text):
    """Speak out loud the given text."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def send_video_feed():
    """Capture video frames from the camera and send to server."""
    video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    video_socket.connect((SERVER_IP, PORT_VIDEO))
    cap = cv2.VideoCapture(0)  # Use the default camera

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Serialize frame
            data = pickle.dumps(frame)
            # Pack the size of the frame and send it first
            message = struct.pack("Q", len(data)) + data
            video_socket.sendall(message)
            time.sleep(0.03)  # Adjust frame rate if necessary
    except Exception as e:
        print(f"Video feed error: {e}")
    finally:
        cap.release()
        video_socket.close()

def send_audio_command():
    """Capture audio commands and send to server."""
    command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    command_socket.connect((SERVER_IP, PORT_COMMAND))
    recognizer = sr.Recognizer()

    try:
        while True:
            with sr.Microphone() as source:
                print("Listening for command...")
                audio = recognizer.listen(source)
            
            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"Recognized command: {command}")
                command_socket.sendall(command.encode('utf-8'))

                # Receive and process the response
                response = command_socket.recv(1024).decode('utf-8')
                print(f"Server response: {response}")
                speak_text(response)  # Speak out the server's response

                # Send servo position if necessary
                if "servo" in response:
                    angle = int(response.split()[-1])  # Example response: "Move servo to 45"
                    servo.turn_angle(angle) # Adjust based on servo controller code

                if command == "exit":
                    break
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
    finally:
        command_socket.close()

if __name__ == "__main__":
    # Start video and audio threads
    threading.Thread(target=send_video_feed, daemon=True).start()
    send_audio_command()
