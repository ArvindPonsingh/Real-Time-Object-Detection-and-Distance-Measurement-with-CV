import numpy as np
import cv2

# Known constants
Known_distance = 30  # Inches
Known_width = 5.7  # Inches (face width for calibration)
thres = 0.5  # Threshold to detect objects
nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress, 0.1 means high suppress

# Colors in BGR format
BLUE = (255, 0, 0)
RED = (0, 0, 255)


font = cv2.FONT_HERSHEY_PLAIN

# Load face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load object detection model
weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Load COCO class names
with open('coco.names', 'r') as f:
    classNames = f.read().strip().split('\n')
print(classNames)

# Focal length finder function
def FocalLength(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

# Distance estimation function (convert to centimeters)
def Distance_finder_v2(focal_length, real_width, perceived_width):
    if perceived_width == 0:  # Avoid division by zero
        return float('inf')
    distance_in_inches = (real_width * focal_length) / perceived_width
    distance_in_centimeters = distance_in_inches * 2.54  # Convert inches to centimeters
    return distance_in_centimeters

# Face detection function
def face_data(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    face_width = 0
    for (x, y, w, h) in faces:
        face_width = w
        cv2.rectangle(image, (x, y), (x+w, y+h), BLUE, 2)
    return face_width, faces

# Reference image
ref_image = cv2.imread("sample.png")
ref_image_face_width, _ = face_data(ref_image)

# Calculate focal length
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
print("Focal Length:", Focal_length_found)

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Object detection
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    for i in indices:
        if isinstance(i, (list, tuple, np.ndarray)):  # Check if `i` is indexable
            i = i[0]

        box = bbox[i]
        x, y, w, h = box
        label = f"{classNames[classIds[i] - 1]}: {int(confs[i] * 100)}%"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 2)
        cv2.putText(frame, label, (x, y-10), font, 2, RED, 3)

        # Estimate distance for objects (using object width as reference)
        if classNames[classIds[i] - 1] == "person":  # Example for detecting "person"
            real_width = 16  # Approximate shoulder width in inches
            perceived_width = w
            distance = Distance_finder_v2(Focal_length_found, real_width, perceived_width)
            cv2.putText(frame, f"Distance:   {int(distance)} cm", (x, y-30), font, 2,RED , 4)  # Increased size and thickness

    # Face detection (no distance text at the top-left)
    face_width, faces = face_data(frame)
    if face_width != 0:
        Distance = Distance_finder_v2(Focal_length_found, Known_width, face_width)
        # Remove the face distance display from the top-left as per request
        # cv2.putText(frame, f"Distance: {int(Distance)} inches", (50, 50), font, 2, ORANGE, 2)

    cv2.imshow("Frame", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
