import requests
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2


server_ip = '192.168.187.70'
class LLaVADescriber:
    def __init__(self, model_name="liuhaotian/LLaVA"):
        """
        Initializes the LLaVA model, processor, and device, and optionally sets up a servo handler.
        Args:
            model_name (str): Path or name of the pretrained LLaVA model.
            servo_handler: An instance of the servo handler for controlling servo movements.
            host (str): Host address for socket connection.
            port (int): Port number for socket connection.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Waiting for connection from Raspberry Pi...")
        print("Connected to Raspberry Pi for image description")

        self.model, self.processor = self.load_llava_model()

    def load_llava_model(self):
        """Loads the LLaVA model and processor from Hugging Face."""
        try:
            processor = BlipProcessor.from_pretrained(self.model_name)
            model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            model.to(self.device).eval()
            message = f"Model '{self.model_name}' loaded on {self.device}."
            requests.post(f"http://{server_ip}:5000/instruction", json={"message": message})
            return model, processor
        except Exception as e:
            message = f"Error loading model '{self.model_name}': {e}"
            requests.post(f"http://{server_ip}:5000/instruction", json={"message": message})
            return None, None
    

    def describe_image(self, image):
        """
        Generates a description for a single image.

        Args:
            image (PIL.Image): Image to describe.

        Returns:
            str: Description generated for the image.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs)
        description = self.processor.decode(output[0], skip_special_tokens=True)
        message = f"Image Description: {description}"
        self.client_socket.sendall(message.encode('utf-8'))
        return description

    def describe_video_with_servo(self, frame_interval=30, servo_step=15):
        """
        Generates descriptions for video frames at regular intervals while rotating the servo.

        Args:
            frame_interval (int): Interval between frames to process.
            servo_step (int): Step size for rotating the servo.

        Returns:
            list: List of descriptions generated for the video frames.
        """
        
        frame_count = 0
        descriptions = []

        for x_angle in range(0, 181, servo_step):
            for y_angle in [0, 180]:  # Rotate Y axis up and down
                # Rotate servo to current angles
                requests.post(f"http://{server_ip}:5000/set_angle", json={"message": f"Servo Position - X: {x_angle}, Y: {y_angle}"})
                # Receive frame from Raspberry Pi
                frame = requests.get(f"http://{server_ip}:5000/get_frame").content
                if frame is None:
                    break

                # Only process every `frame_interval`-th frame
                if frame_count % frame_interval == 0:
                    # Convert the frame to PIL format for processing
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    description = self.describe_image(image)
                    descriptions.append((frame_count, x_angle, y_angle, description))
                    message = f"Frame {frame_count}, X angle {x_angle}, Y angle {y_angle}: {description}"
                    requests.post(f"http://{server_ip}:5000/instruction", json={"message": message})   

                    # Send servo position to Raspberry Pi
                    servo_position_message = f"Servo Position - X: {x_angle}, Y: {y_angle}"
                    requests.post(f"http://{server_ip}:5000/set_angle", json={"message": servo_position_message})

                frame_count += 1

        return descriptions


# Example usage
if __name__ == "__main__":
    model_name = "Salesforce/blip-image-captioning-base"

   # Assuming you have a ServoPositionHandler class
    describer = LLaVADescriber(model_name)
    try:
        descriptions = describer.describe_video_with_servo()  # Process video frames from Raspberry Pi
    except KeyboardInterrupt:
       