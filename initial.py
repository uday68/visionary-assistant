from flask import request
from describe import LLaVADescriber
import requests as request
server_ip = "192.168.187.70"
def initial_scene():
           # Replace with the actual port number  # Assuming you have a ServoPositionHandler class
    describer = LLaVADescriber(model_name="Salesforce/blip-image-captioning-base")
    
    try:
        descriptions = describer.describe_video_with_servo()  # Process video frames from Raspberry Pi
        request.post(f"http://{server_ip}:{5000}/instruction", json={"message": "Video description completed."})
    except KeyboardInterrupt:
        describer.close_connection()
        request.post(f"http://{server_ip}:{5000}/instruction", json={"message": "Video description interrupted."})
if __name__ == "__main__":
    initial_scene()