"""
Author: Tech With Haberjame
GitHub: Tech-With-Haberjame
Tiktok: haberjame
Contact: haberjame.tech@gmail.com
"""
import cv2
import mediapipe as mp

# Initialize the face detection module
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Access the webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 2560)  # Set width
video_capture.set(4, 1440)  # Set height

# Set luxury design colors
box_color = (0, 191, 255)  # Light Blue
box_thickness = 3

while True:
    # Read the current frame from the webcam
    ret, frame = video_capture.read()

    # Convert the image to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection on the current frame
    results = face_detection.process(frame_rgb)

    # Extract bounding box information for each detected face
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            xmax = int((bbox.xmin + bbox.width) * w)
            ymax = int((bbox.ymin + bbox.height) * h)

            # Draw bounding box on the frame with luxury design
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, box_thickness)

    # Display the frame with bounding boxes
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()