import cv2
import signal
import sys
from crisp_py.camera import Camera, CameraConfig

def signal_handler(sig, frame):
    print("Shutting down camera...")
    camera.shutdown()
    cv2.destroyAllWindows()
    sys.exit(0)

camera_config = CameraConfig(
    camera_name="primary",
    resolution=(480, 640),  
    camera_color_image_topic="/dave/color/image_raw",  
    camera_color_info_topic="/dave/color/camera_info",
)

camera = Camera(config=camera_config)  
signal.signal(signal.SIGINT, signal_handler)
camera.wait_until_ready() 

# Convert the image from RGB to BGR for correct color display in OpenCV
image_bgr = cv2.cvtColor(camera.current_image, cv2.COLOR_RGB2BGR)

cv2.imshow("Camera Image", image_bgr)
cv2.waitKey(0)

