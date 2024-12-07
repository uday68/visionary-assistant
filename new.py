
import cv2


def generate_frames():
    camera = cv2.VideoCapture('http://192.168.187.70:5000/video_feed')  # Use 0 for USB webcam or /dev/video0
    while True:
        success, frame = camera.read()
        if not success:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

generate_frames()
  