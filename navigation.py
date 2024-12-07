import logging
import torch
import cv2
import numpy as np
from torchvision import transforms
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import pyttsx3
import warnings
import threading
import queue
import time

# Suppress specific warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

# Constants
DEPTH_SCALING_FACTOR = 1000.0  # Scaling factor for depth values
CLOSE_DISTANCE_THRESHOLD = 1.0
WARNING_DISTANCE_THRESHOLD = 2.0
GRID_ROWS, GRID_COLS = 10, 10
FRAME_WIDTH, FRAME_HEIGHT = 680, 480
GRID_ROWS_DETAIL, GRID_COLS_DETAIL = 16, 16
CELL_WIDTH = FRAME_WIDTH / GRID_COLS_DETAIL
CELL_HEIGHT = FRAME_HEIGHT / GRID_ROWS_DETAIL

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Configure logging
logging.basicConfig(level=logging.INFO)  # Adjust level as needed
logger = logging.getLogger(__name__)

class DetectronModel:
    def __init__(self, device='cpu'):
        """
        Initialize the Detectron2 model for panoptic segmentation.
        :param device: 'cpu' or 'cuda' depending on available hardware.
        """
        self.predictor = self.init_model(device)
        self.metadata = MetadataCatalog.get("coco_2017_panoptic_val")  # Correct metadata

    def init_model(self, device):
        """
        Configure the panoptic segmentation model.
        :param device: 'cpu' or 'cuda' depending on available hardware.
        :return: DefaultPredictor object for inference.
        """
        cfg = get_cfg()
        cfg.MODEL.DEVICE = device
        # Use a valid panoptic segmentation configuration file from Model Zoo
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for instance predictions
        return DefaultPredictor(cfg)

    def detect_objects(self, frame):
        """
        Perform panoptic segmentation on the input frame.
        :param frame: Input image (NumPy array).
        :return: Detected bounding boxes, class names, panoptic segmentation map, and segment info.
        """
        outputs = self.predictor(frame)

        # Check for instance predictions
        if "instances" in outputs and outputs["instances"].has("pred_classes"):
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            class_ids = outputs["instances"].pred_classes.cpu().numpy()
            class_names = [self.metadata.thing_classes[class_id] for class_id in class_ids]
        else:
            boxes, class_names = [], []

        # Panoptic segmentation output
        if "panoptic_seg" in outputs:
            panoptic_output, segments_info = outputs["panoptic_seg"]
        else:
            panoptic_output, segments_info = None, []

        return boxes, class_names, panoptic_output, segments_info
class EdgeTracker:
    def __init__(self, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, cell_width=CELL_WIDTH, cell_height=CELL_HEIGHT):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.edge_grid = np.empty((self.grid_rows, self.grid_cols), dtype=object)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Vectorized processing for grid cells
        for row in range(self.grid_rows):
            y_start = int(row * self.cell_height)
            y_end = y_start + int(self.cell_height)
            for col in range(self.grid_cols):
                x_start = int(col * self.cell_width)
                x_end = x_start + int(self.cell_width)
                cell_edges = edges[y_start:y_end, x_start:x_end]
                if cell_edges.size:
                    self.edge_grid[row, col] = cell_edges
                    overlay = cv2.cvtColor(cell_edges, cv2.COLOR_GRAY2BGR)
                    frame[y_start:y_end, x_start:x_end] = cv2.addWeighted(
                        frame[y_start:y_end, x_start:x_end],
                        0.7,
                        overlay,
                        0.3,
                        0
                    )
        return frame

class GridManager:
    def __init__(self, grid_rows=GRID_ROWS, grid_cols=GRID_COLS):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

    def update_grid(self, depth_map):
        """
        Updates the grid based on the aggregated depth_map.
        Aggregates the depth_map to match the grid size by averaging.
        """
        # Ensure depth_map is a numpy array
        depth_map = np.array(depth_map)

        # Calculate the size of each cell in the depth_map
        depth_rows, depth_cols = depth_map.shape
        cell_height = depth_rows // self.grid_rows
        cell_width = depth_cols // self.grid_cols

        # Initialize aggregated depth_map
        aggregated_depth = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                y_start = row * cell_height
                y_end = (row + 1) * cell_height if row < self.grid_rows - 1 else depth_rows
                x_start = col * cell_width
                x_end = (col + 1) * cell_width if col < self.grid_cols - 1 else depth_cols
                cell_depth = depth_map[y_start:y_end, x_start:x_end]
                aggregated_depth[row, col] = cell_depth.mean()

        # Apply thresholds
        blocked = aggregated_depth < CLOSE_DISTANCE_THRESHOLD
        warning = (aggregated_depth >= CLOSE_DISTANCE_THRESHOLD) & (aggregated_depth < WARNING_DISTANCE_THRESHOLD)

        self.grid[blocked] = 2  # Green for blocked
        self.grid[warning] = 1  # Yellow for warning
        self.grid[~(blocked | warning)] = 0  # White for clear

    def get_grid_status(self):
        return self.grid

class DepthEstimator:
    def __init__(self, device='cpu'):
        self.device = device
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device)
            self.model.eval()
            logger.info("MiDaS model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise

        # Define normalization parameters as tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def estimate_depth(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (384, 384))
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
            img_tensor = torch.from_numpy(frame_transposed).unsqueeze(0).to(self.device)
            img_tensor = (img_tensor - self.mean) / self.std

            with torch.no_grad():
                depth_map = self.model(img_tensor).squeeze().cpu().numpy()
            depth_map_resized = self.resize_depth_map(depth_map, frame.shape[:2])
            logger.debug("Depth estimation successful.")
            return depth_map_resized
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None

    @staticmethod
    def resize_depth_map(depth_map, target_size):
        if depth_map is None:
            raise ValueError("Received None for depth_map")
        return cv2.resize(depth_map, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def calibrate_depth(depth_value):
        return depth_value / DEPTH_SCALING_FACTOR

class AlertManager:
    def __init__(self):
        self.alert_queue = queue.Queue()
        self.alert_thread = threading.Thread(target=self.process_alerts, daemon=True)
        self.alert_thread.start()

    def provide_alert(self, depth, position_x, frame_width, class_name):
        alert_message = ""
        movement_suggestion = ""

        if position_x < frame_width * 0.3:
            position = "left"
            movement_suggestion = "Please move a little to the right."
        elif position_x > frame_width * 0.7:
            position = "right"
            movement_suggestion = "Please move a little to the left."
        else:
            position = "center"

        if class_name == "road":
            if depth < CLOSE_DISTANCE_THRESHOLD:
                alert_message = "Obstacle very close on the road. Please stop and proceed carefully."
            elif depth < WARNING_DISTANCE_THRESHOLD:
                alert_message = "Warning: Obstacle ahead on the road. Proceed with caution."
        elif class_name in ["pedestrian", "vehicle"]:
            if depth < CLOSE_DISTANCE_THRESHOLD:
                alert_message = f"Obstacle very close on your {position}. {movement_suggestion} Please stop and proceed carefully."
            elif depth < WARNING_DISTANCE_THRESHOLD:
                alert_message = f"Warning: Obstacle ahead on your {position}. {movement_suggestion} Proceed with caution."
        elif class_name == 'stairs':
            if depth < WARNING_DISTANCE_THRESHOLD:
                alert_message = "You are approaching the stairs."
            elif depth < CLOSE_DISTANCE_THRESHOLD:
                alert_message = "You are very close to the stairs."

        if alert_message:
            logger.info(alert_message)
            self.alert_queue.put(alert_message)

    def process_alerts(self):
        while True:
            try:
                message = self.alert_queue.get()
                if message:
                    self.speak_alert(message)
                self.alert_queue.task_done()
            except Exception as e:
                logger.error(f"Error in alert processing thread: {e}")

    def speak_alert(self, message):
        try:
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")

class NavigationSystem:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        self.predictor = DetectronModel(device=device)  # Use GPU if available
        self.depth_estimator = DepthEstimator(device=device)
        self.grid_manager = GridManager()
        self.alert_manager = AlertManager()
        self.edge_tracker = EdgeTracker()
        self.frame_skip = 2
        self.alert_cooldown = 2  # seconds
        self.last_alert_time = 0

    def process_frame(self, frame):
        logger.debug("Processing frame")
        depth_map = self.depth_estimator.estimate_depth(frame)
        if depth_map is None:
            logger.error("Depth map is None.")
            return frame
        logger.debug(f"Depth map shape: {depth_map.shape}")

        boxes, class_names, panoptic_output, segments_info = self.predictor.detect_objects(frame)
        logger.debug(f"Detected {len(boxes)} objects.")

        self.grid_manager.update_grid(depth_map)
        grid_status = self.grid_manager.get_grid_status()

        blocked = self.is_path_blocked(panoptic_output, segments_info, frame.shape[1], frame.shape[0])
        surface_info = self.analyze_surface(depth_map)

        frame = self.overlay_grid_status(frame, blocked, surface_info)
        return frame

    def handle_path_blocked(self, segment_id):
        messages = {
            1: "Person detected. Path blocked. Ask the person to move.",
            2: "Object detected. Path blocked. Remove the object.",
            3: "Car detected. Path blocked by car."
        }
        message = messages.get(segment_id, "Path is blocked.")
        logger.info(message)
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            self.alert_manager.provide_alert(None, 0, FRAME_WIDTH, message)  # Adjust parameters as needed
            self.last_alert_time = current_time

    def is_path_blocked(self, panoptic_output, segments_info, frame_width, frame_height):
        """
        Determines if the path is blocked based on panoptic segmentation output.
        """
        if panoptic_output is None or segments_info is None:
            return False

        # Define path region (e.g., center third of the frame horizontally and lower half vertically)
        path_x_start = int(frame_width * 0.3)
        path_x_end = int(frame_width * 0.7)
        path_y_start = int(frame_height * 0.5)
        path_y_end = frame_height

        # Extract mask and segments
        masks = panoptic_output[0].cpu().numpy()
        segments = segments_info

        for segment in segments:
            class_id = segment["category_id"]
            # Define obstacle classes based on COCO dataset
            obstacle_classes = [1, 2, 3, 4, 6, 8]  # person, bicycle, car, motorcycle, bus, truck
            if class_id in obstacle_classes:
                mask = masks == segment["id"]
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                # Check if any part of the object is within the path region
                if (xs >= path_x_start).any() and (xs <= path_x_end).any() and (ys >= path_y_start).any():
                    # Optionally, retrieve depth information for the specific area
                    # Here we assume depth_map is accessible; adjust as needed
                    # For demonstration, we'll trigger an alert
                    self.handle_path_blocked(self.get_category_id(segment["category_name"]))
                    return True
        return False

    @staticmethod
    def get_category_id(class_name):
        mapping = {
            "person": 1,
            "bicycle": 2,
            "car": 3,
            "motorcycle": 4,
            "bus": 6,
            "truck": 8
        }
        return mapping.get(class_name, 0)

    def analyze_surface(self, depth_map):
        gradient_x, gradient_y = np.gradient(depth_map)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return {
            "elevation": np.mean(depth_map),
            "patchiness": np.std(depth_map),
            "incline": np.mean(gradient_x),
            "decline": np.mean(gradient_y),
            "total_incline": np.mean(gradient_magnitude)
        }

    def overlay_grid_status(self, frame, blocked, surface_info):
        # Get aggregated depth grid
        grid_status = self.grid_manager.get_grid_status()

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x_start = int(col * CELL_WIDTH)
                y_start = int(row * CELL_HEIGHT)
                x_end = x_start + int(CELL_WIDTH)
                y_end = y_start + int(CELL_HEIGHT)

                depth_value = grid_status[row, col]
                if depth_value == 2:
                    color = (0, 255, 0)  # Green for blocked
                elif depth_value == 1:
                    color = (0, 255, 255)  # Yellow for warning
                else:
                    color = (255, 255, 255)  # White for clear

                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)

        if blocked:
            cv2.putText(frame, "Path Blocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display surface information
        cv2.putText(frame, f"Elevation: {surface_info['elevation']:.2f}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Patchiness: {surface_info['patchiness']:.2f}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Incline: {surface_info['incline']:.2f}", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Decline: {surface_info['decline']:.2f}", (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def run(self):
        try:
            cap = cv2.VideoCapture(0)  # Replace with your video stream
            if not cap.isOpened():
                logger.error("Cannot open video stream.")
                return

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to grab frame.")
                    break

                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue

                # Resize frame for faster processing
                frame_resized = cv2.resize(frame, (320, 240))

                # Edge tracking
                processed_frame = self.edge_tracker.process_frame(frame_resized)

                # Process frame
                processed_frame = self.process_frame(processed_frame)

                # Display the processed frame
                cv2.imshow('Navigation System', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_count += 1

        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    nav = NavigationSystem()
    nav.run()