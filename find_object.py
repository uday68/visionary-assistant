# Description: ObjectFinder class to detect objects using YOLO model and servo control.
import cv2
import requests
from ultralytics import YOLO
import numpy as np
import time
server_ip = "192.168.187.70"
class ObjectFinder:
    def __init__(self, host, port, class_names):
        self.model = YOLO('./model/yolov8x-oiv7.pt')  # Load custom weights if necessary
        
        # Store class names
        self.class_names = class_names
        
        # Camera and scanning parameters
        self.camera_fov = 60  # Camera field of view in degrees
        self.frame_center_x = 320
        self.frame_center_y = 240
        self.camera_height = 1.5
        self.x_angles = range(0, 176, 15)  # X-axis: 0° to 175° in steps
        self.y_angles = range(0, 136, 45)  # Y-axis: 0° to 135° in steps

        requests.post(f"http://{server_ip}:5000/instruction",{"message":"Object Finder Initialized"})



    @staticmethod
    def load_class_names(filename="classes.txt"):
        with open(filename, "r") as file:
            class_names = [line.strip() for line in file.readlines()]
        return class_names

    def receive_frame(self):
        """Receive a frame from the Raspberry Pi camera server."""
        try:
            frame = requests.get(f"http://{server_ip}:5000/get_frame").content
            frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
        except requests.exceptions.RequestException as e:   
            print(f"Error receiving frame: {e}")
        return frame

    def send_servo_position(self, x_angle, y_angle):
        """Send the servo position to the Raspberry Pi for object search."""
        try:
            requests.post(f"http://{server_ip}:5000/set_angle", json={"x_angle": x_angle, "y_angle": y_angle})
        except  :
            print(f"Error sending servo position: ")

    def detect_object(self, frame, target_class_id):
        """Detect specified objects in the frame using YOLO model."""
        results = self.model(frame)
        detected_boxes = []
        
        for detection in results.xyxy[0]:  # Iterate over detected objects
            x1, y1, x2, y2, conf, class_id = detection[:6]
            if conf > 0.7 and int(class_id) == target_class_id:
                detected_boxes.append((int(x1), int(y1), int(x2), int(y2), int(class_id)))

        return detected_boxes

    def find_object_with_servo(self, object_name):
        """Scan for an object by moving the servo in both X and Y directions."""
        target_class_id = self.get_class_id(object_name)
        if target_class_id is None:
            return
        
        for y_angle in self.y_angles:
            for x_angle in self.x_angles:
                # Send servo position to Raspberry Pi
                self.send_servo_position(x_angle, y_angle)
                
                # Wait briefly for the servo to adjust before capturing frame
                time.sleep(0.5)  
                
                # Capture frame and attempt to detect object
                frame = self.receive_frame()
                detected_boxes = self.detect_object(frame, target_class_id)
                
                if detected_boxes:
                    # If detected, stop scanning and announce location
                    x1, y1, x2, y2, _ = detected_boxes[0]
                    center_x = (x1 + x2) // 2
                    distance, angle = self.calculate_distance_and_angle(center_x)
                    message = f"Detected {object_name} at {distance:.2f} meters, {angle:.2f} degrees."
                    self.send_message(message)
                    return  # Stop searching once object is found

        # If not detected, inform the user
        self.send_message(f"Could not detect {object_name} in the scanned area.")

    def calculate_distance_and_angle(self, center_x):
        """Calculate distance and angle to the object."""
        delta_x = center_x - self.frame_center_x
        angle = (delta_x / self.frame_center_x) * (self.camera_fov / 2)
        distance = self.camera_height / np.tan(np.deg2rad(angle + 1e-6))
        return distance, angle

    def send_message(self, message):
        try:
            requests.post(f"http://{server_ip}:5000/instruction",{"message":message})
        except :
            print(f"Error sending message: ")


# Example usage
if __name__ == "__main__":
    class_names = ObjectFinder.load_class_names()# Load class names from file
