import datetime
import requests
import wikipedia
import logging
import cv2
from find_object import ObjectFinder
from describe import ImageDescriber
from navigation import Navigation
import facedetection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Server IP address
SERVER_IP = '192.168.187.70'

# Initialize modules
object_finder = ObjectFinder()
describer = ImageDescriber(server_ip=SERVER_IP)
nav_system = Navigation(server_ip=SERVER_IP)

def send_feedback(message):
    """Send feedback message to the server to be spoken."""
    try:
        response = requests.post(f'http://{SERVER_IP}:5000/speak', json={'text': message})
        if response.status_code == 200:
            logging.info("Feedback sent successfully.")
        else:
            logging.error(f"Failed to send feedback: {response.status_code}")
    except Exception as e:
        logging.error(f"Error sending feedback: {e}")

def handle_command(command):
    """Receive commands from Raspberry Pi and execute the respective action."""
    try:
        if 'find object' in command:
            response = perform_object_detection()
            send_feedback(response)
        elif 'describe' in command:
            description = describe_scenario()
            send_feedback(description)
        elif 'navigate' in command:
            nav_system.run()
            send_feedback("Navigation complete.")
        elif "search" in command:
            search_query = command.split("search", 1)[1].strip()
            try:
                search_result = wikipedia.summary(search_query, sentences=2)
                send_feedback(search_result)
            except wikipedia.exceptions.DisambiguationError:
                send_feedback("Ambiguous search query. Please be more specific.")
            except wikipedia.exceptions.PageError:
                send_feedback("No search results found.")
        elif "help" in command:
            help_text = ("Available commands:\n"
                         "find object - Perform object detection\n"
                         "describe - Describe the current scenario\n"
                         "navigate - Navigate to a destination\n"
                         "search <query> - Search Wikipedia for information\n"
                         "help - Display this help message\n"
                         "date - Get current date\n"
                         "time - Get current time\n"
                         "exit - Exit the program")
            send_feedback(help_text)
        elif 'date' in command:
            send_feedback(datetime.datetime.now().strftime("%Y-%m-%d"))
        elif 'time' in command:
            send_feedback(datetime.datetime.now().strftime("%H:%M:%S"))
        elif 'exit' in command:
            send_feedback("Goodbye!")
            return False
        else:
            send_feedback("Unknown command.")
        return True
    except Exception as e:
        logging.error(f"Error handling command: {e}")
        return True

def perform_object_detection():
    """Invoke object finder for detection."""
    detected_object = object_finder.find_and_announce_object()
    return detected_object if detected_object else "No object detected."

def describe_scenario():
    """Describe the current scene using a frame from the video feed."""
    frame, description, angle = describer.describe_video_with_servo()
    return description

def main():
    try:
        while True:
            # Example command input (replace with actual command input mechanism)
            command = input("Enter command: ")
            if not handle_command(command):
                break
    except KeyboardInterrupt:
        logging.info("Program interrupted by user.")
    except Exception as e:
        logging.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    main()