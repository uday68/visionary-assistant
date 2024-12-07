import pyttsx3
import time
import numpy as np
import cv2
import requests
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

class RoadSafety:
    def __init__(self, person, servo_handler):
        self.person = person
        self.servo_handler = servo_handler
        self.engine = pyttsx3.init()
        self.grid = np.zeros((3, 3))  # Lane grid for left, center, right lanes

        # Initialize Cityscapes segmentation model
        self.predictor = self.init_cityscapes_model()
        self.metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val")

    def init_cityscapes_model(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        return DefaultPredictor(cfg)

    def detect_objects(self, frame):
        outputs = self.predictor(frame)
        boxes = outputs["instances"].pred_boxes if outputs["instances"].has("pred_boxes") else []
        class_ids = outputs["instances"].pred_classes if outputs["instances"].has("pred_classes") else []
        class_names = [self.metadata.thing_classes[class_id] for class_id in class_ids]
        return boxes, class_names

    def detect_vehicle(self, frame):
        boxes, class_names = self.detect_objects(frame)
        vehicles = [box for box, name in zip(boxes, class_names) if name in ["car", "truck", "bus", "bicycle", "motorcycle"]]
        return vehicles

    def detect_traffic_light(self, frame):
        boxes, class_names = self.detect_objects(frame)
        traffic_lights = [box for box, name in zip(boxes, class_names) if name == "traffic light"]
        return traffic_lights

    def get_traffic_light_color(self, frame, traffic_light_box):
        x1, y1, x2, y2 = traffic_light_box
        cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for red, yellow, green
        red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([50, 100, 100]), np.array([70, 255, 255]))

        red_area, yellow_area, green_area = map(np.sum, [red_mask, yellow_mask, green_mask])

        # Determine light color
        if red_area > yellow_area and red_area > green_area:
            return "red"
        elif yellow_area > red_area and yellow_area > green_area:
            return "yellow"
        elif green_area > red_area and green_area > yellow_area:
            return "green"
        else:
            return "unknown"

    def update_lane_grid(self, vehicles, frame_width):
        self.grid.fill(0)  # Reset grid for each frame
        lane_width = frame_width // 3
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle
            center_x = (x1 + x2) // 2
            lane = center_x // lane_width
            if lane < 3:
                self.grid[lane][1] = 1  # Mark vehicle presence in the lane

    def check_vehicle_status(self):
        alerts = []
        if self.grid[0][1] == 1:
            alerts.append("Vehicle in left lane.")
        if self.grid[1][1] == 1:
            alerts.append("Vehicle in center lane.")
        if self.grid[2][1] == 1:
            alerts.append("Vehicle in right lane.")
        for alert in alerts:
            self.engine.say(alert)
        if not alerts:
            self.engine.say("No vehicles detected. Safe to cross.")
        self.engine.runAndWait()

    def navigate_to_zebra_crossing(self, frame):
        crossings = [box for box, name in zip(*self.detect_objects(frame)) if name == "road"]
        if crossings:
            self.engine.say("Zebra crossing detected. Proceed.")
            self.servo_handler.set_angle([90, 90, 90])
        else:
            self.engine.say("No zebra crossing detected. Please wait.")
        self.engine.runAndWait()

    def detect_and_control_traffic_light(self, frame):
        traffic_lights = self.detect_traffic_light(frame)
        for light_box in traffic_lights:
            light_color = self.get_traffic_light_color(frame, light_box)
            if light_color == "green":
                self.engine.say("Green light. Safe to cross.")
            elif light_color == "red":
                self.engine.say("Red light. Please wait.")
            else:
                self.engine.say("No clear traffic light signal. Please proceed cautiously.")
        self.engine.runAndWait()

    def run_road_safety_protocol(self, frame):
        self.detect_and_control_traffic_light(frame)
        vehicles = self.detect_vehicle(frame)
        self.update_lane_grid(vehicles, frame.shape[1])
        self.check_vehicle_status()
        self.navigate_to_zebra_crossing(frame)
    def run(self):
        while True and self.detect_objects() in ["car", "truck", "bus", "bicycle", "motorcycle"]:
            response = requests.get('http://get_frame')
            if response.status_code == 200:
                frame = response.content
                self.run_road_safety_protocol(frame)
            if response.status_code == 404:
                msg = "Error: Camera feed not available."
                requests.post('http://send_message', json={'message': msg})
                requests.post('http://send_email', json={'message':"retrying"})
                time.sleep(5)
                self.run()  # Retry
            time.sleep(1)
            requests.post('http://send_message', json={'message': "Road safety protocol completed."})
            requests.post('http://resume_navigation')
            return

            