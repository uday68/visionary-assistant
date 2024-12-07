import socket
import cv2
import pickle
import struct
import pytesseract
import pyttsx3
import mediapipe as mp
from servo_control import ServoPositionHandler

class OCRNavigationSystem:
    def __init__(self):
        self.text_to_speech = pyttsx3.init()

    def process_frame(self, frame):
        """
        Processes a single frame to detect text using Tesseract OCR.

        Args:
            frame (np.ndarray): The captured frame from the camera.

        Returns:
            str: Detected text from the frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_text = pytesseract.image_to_string(gray)
        self.text_to_speech.say(detected_text)
        self.text_to_speech.runAndWait()
        return detected_text

class HandTracker:
    def __init__(self, servo_handler):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        self.servo_handler = servo_handler

    def process_frame(self, frame):
        """
        Processes a single frame to detect hands using MediaPipe and moves the servo.

        Args:
            frame (np.ndarray): The captured frame from the camera.

        Returns:
            np.ndarray: Frame with hand landmarks drawn.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # Move servo based on hand position
                x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y
                self.move_servo(x, y)

        return frame

    def move_servo(self, x, y):
        """
        Moves the servo based on the hand position.

        Args:
            x (float): X-coordinate of the hand (0 to 1).
            y (float): Y-coordinate of the hand (0 to 1).
        """
        # Map x and y to servo angles (-1 to 1)
        servo_x = 2 * x - 1  # Map x from [0, 1] to [-1, 1]
        servo_y = 2 * y - 1  # Map y from [0, 1] to [-1, 1]
        self.servo_handler.set_angle(servo_x)  # Example: move servo based on x-coordinate

def receive_frame(conn):
    """
    Receives a frame from the client.

    Args:
        conn (socket.socket): The socket connection to the client.

    Returns:
        np.ndarray: The received frame.
    """
    data = b""
    payload_size = struct.calcsize("Q")
    while len(data) < payload_size:
        packet = conn.recv(4 * 1024)  # 4K
        if not packet:
            return None
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4 * 1024)

    frame_data = data[:msg_size]
    frame = pickle.loads(frame_data)
    return frame

def server_program():
    """
    Sets up the server to receive frames from the client, process them, and send back responses.
    """
    HOST = '0.0.0.0'  # Listen on all available interfaces
    PORT = 12345

    ocr_system = OCRNavigationSystem()
    servo_handler = ServoPositionHandler(pin=17)  # Replace with the actual GPIO pin
    hand_tracker = HandTracker(servo_handler)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print("Waiting for connection from client...")
        conn, addr = s.accept()
        print(f"Connected by {addr}")

        with conn:
            try:
                while True:
                    frame = receive_frame(conn)
                    if frame is None:
                        break

                    # Process frame for hand tracking and servo control
                    frame_with_hands = hand_tracker.process_frame(frame)

                    # Process frame for OCR
                    detected_text = ocr_system.process_frame(frame_with_hands)
                    response_message = f"Detected Text: {detected_text}"
                    conn.sendall(response_message.encode('utf-8'))

            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    server_program()