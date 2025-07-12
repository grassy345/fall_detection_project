# -*- coding: utf-8 -*-
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
    cap = cv2.VideoCapture("stock videos/1.mp4")
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

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    # Convert to numpy arrays
    a = np.array([point1.x, point1.y])
    b = np.array([point2.x, point2.y])
    c = np.array([point3.x, point3.y])
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def get_key_angles(landmarks):
    """Extract key body angles for fall detection"""
    # Get relevant landmark points
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
    
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
    
    # Calculate key angles
    # Torso angle (shoulder-hip-knee)
    left_torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_torso_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    
    # Leg angle (hip-knee-ankle)
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    return {
        'left_torso': left_torso_angle,
        'right_torso': right_torso_angle,
        'left_leg': left_leg_angle,
        'right_leg': right_leg_angle
    }

def display_angles_on_frame(frame, angles):
    """Display calculated angles on the frame for debugging"""
    if angles:
        y_offset = 30
        cv2.putText(frame, f"Left Torso: {angles['left_torso']:.1f} deg", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Right Torso: {angles['right_torso']:.1f} deg", 
                   (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Left Leg: {angles['left_leg']:.1f} deg", 
                   (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Right Leg: {angles['right_leg']:.1f} deg", 
                   (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def detect_fall_by_angles(angles):
    """
    Detect falls based on body angles
    Returns: (is_fall_detected, fall_reason)
    """
    if not angles:
        return False, "No pose detected"
    
    # Define thresholds (you'll need to tune these)
    TORSO_FALL_THRESHOLD = 120  # degrees - if torso is too bent
    LEG_BENT_THRESHOLD = 140    # degrees - if legs are too bent
    
    # Get average angles for more stability
    avg_torso = (angles['left_torso'] + angles['right_torso']) / 2
    avg_leg = (angles['left_leg'] + angles['right_leg']) / 2
    
    # Fall detection logic
    fall_detected = False
    reason = ""
    
    # Check for severely bent torso (person bending over or falling forward)
    if avg_torso < TORSO_FALL_THRESHOLD:
        fall_detected = True
        reason = f"Bent torso detected: {avg_torso:.1f} deg"
    
    # Check for severely bent legs (person crouching/sitting on ground)
    elif avg_leg < LEG_BENT_THRESHOLD:
        fall_detected = True
        reason = f"Bent legs detected: {avg_leg:.1f} deg"
    
    # Check for asymmetric poses (one side very different from other)
    torso_diff = abs(angles['left_torso'] - angles['right_torso'])
    if torso_diff > 50:  # Large difference between sides
        fall_detected = True
        reason = f"Asymmetric pose: {torso_diff:.1f} deg difference"
    
    return fall_detected, reason

def display_fall_alert(frame, is_fall, reason):
    """Display fall alert or normal status at the bottom of the frame"""
    y_base = frame.shape[0] - 60
    if is_fall:
        # Red text at the bottom
        cv2.putText(frame, "FALL DETECTED!", (10, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(frame, reason, (10, y_base + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Green status at the bottom
        cv2.putText(frame, "Status: Normal", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
            # End of video reached, reset to first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Process frame for pose detection
        processed_frame, pose_results = process_frame(frame, pose)

        # Draw pose landmarks if detected
        if pose_results.pose_landmarks:
            draw_pose_landmarks(processed_frame, pose_results, mp_drawing, mp_pose)
            angles = get_key_angles(pose_results.pose_landmarks.landmark)
            display_angles_on_frame(processed_frame, angles)
            
            # Detect fall and display alert/status
            is_fall, reason = detect_fall_by_angles(angles)
            display_fall_alert(processed_frame, is_fall, reason)
        else:
            # If no pose detected, show normal status
            display_fall_alert(processed_frame, False, "No pose detected")

        # Concatenate original and processed frames horizontally
        combined = np.hstack((frame, processed_frame))
        cv2.imshow('Fall Detection System (Original | Processed)', combined)
        
        # Check for quit condition (keyboard or window closed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Fall Detection System (Original | Processed)', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Handle cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Fall Detection System Stopped.")

if __name__ == "__main__":
    main()