# cv2 for image processing and manipulation
# mediapipe = MediaPipe framework for pose estimation
import cv2
import mediapipe as mp
import numpy as np

def initialize_mediapipe():
    """Initialize MediaPipe pose detection"""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Create the pose detector object
    pose = mp_pose.Pose(
        static_image_mode=False,    # For video stream
        model_complexity=1,         # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,      # Smooth between frames
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return pose, mp_drawing, mp_pose

def setup_camera():
    """Initialize camera capture"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def process_frame(frame, pose_detector):
    """Process a single frame for pose detection"""
    # MediaPipe requires RGB format, OpenCV uses BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run pose detection
    pose_results = pose_detector.process(rgb_frame)
    
    # Convert back to BGR for OpenCV display
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    return bgr_frame, pose_results

def draw_pose_landmarks(frame, pose_results, drawing_utils, mp_pose):
    """Draw skeleton on the frame"""
    if pose_results.pose_landmarks:
        # Draw the pose landmarks
        drawing_utils.draw_landmarks(
            frame, 
            pose_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(255,0,0), thickness=2)
        )

def main():
    """Main program loop"""
    # Initialize everything
    pose, mp_drawing, mp_pose = initialize_mediapipe()
    cap = setup_camera()
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Fall Detection System Started. Press 'q' to quit.")

    # Main camera loop
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame for pose detection
        processed_frame, pose_results = process_frame(frame, pose)

        # Draw pose landmarks if detected
        if pose_results.pose_landmarks:
            draw_pose_landmarks(processed_frame, pose_results, mp_drawing, mp_pose)

        # Concatenate original and processed frames horizontally
        combined = np.hstack((frame, processed_frame))
        cv2.imshow('Fall Detection System (Original | Processed)', combined)
        
        # Check for quit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Handle cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Fall Detection System Stopped.")

if __name__ == "__main__":
    main()