# -*- coding: utf-8 -*-
# cv2 for image processing and manipulation
# mediapipe = MediaPipe framework for pose estimation
import cv2
import mediapipe as mp
import numpy as np
import time

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
    cap = cv2.VideoCapture("stock videos/sinan/14_rotated_resized.mp4")
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

# def get_vertical_positions(landmarks):
#     """
#     Extract vertical positions (y-coordinates) of key landmarks for fall detection
#     Returns dictionary with normalized y values (0.0 = top, 1.0 = bottom)
#     """
#     # Get nose landmark (head position)
#     nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
    
#     # Get hip landmarks
#     left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
#     right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    
#     # Calculate hip center y-coordinate
#     hip_center_y = (left_hip.y + right_hip.y) / 2
    
#     # Get shoulder landmarks (for optional use)
#     left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
#     right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
#     shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    
#     # Get knee landmarks (for optional use)
#     left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
#     right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
#     knee_center_y = (left_knee.y + right_knee.y) / 2
    
#     return {
#         'nose_y': nose.y,
#         'hip_y': hip_center_y,
#         'shoulder_y': shoulder_center_y,
#         'knee_y': knee_center_y
#     }

def update_landmark_history(landmarks, history, max_frames):
    """Store current frame landmarks and maintain history size"""
    # Extract key landmark positions (we'll track hip and shoulder centers)
    left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    # Calculate center points
    hip_center_x = (left_hip.x + right_hip.x) / 2
    hip_center_y = (left_hip.y + right_hip.y) / 2
    
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # Store as dictionary
    frame_data = {
        'hip': (hip_center_x, hip_center_y),
        'shoulder': (shoulder_center_x, shoulder_center_y)
    }
    
    # Add to history
    history.append(frame_data)
    
    # Keep only the last max_frames
    if len(history) > max_frames:
        history.pop(0)  # Remove oldest frame

def calculate_velocity(history):
    """Calculate velocity from landmark history"""
    if len(history) < 2:
        return None  # Need at least 2 frames
    
    # Get the oldest and newest frames
    oldest_frame = history[0]
    newest_frame = history[-1]
    
    # Calculate displacement for hip
    hip_dx = newest_frame['hip'][0] - oldest_frame['hip'][0]
    hip_dy = newest_frame['hip'][1] - oldest_frame['hip'][1]
    
    # Calculate displacement for shoulder
    shoulder_dx = newest_frame['shoulder'][0] - oldest_frame['shoulder'][0]
    shoulder_dy = newest_frame['shoulder'][1] - oldest_frame['shoulder'][1]
    
    # Calculate speeds (distance moved)
    hip_speed = np.sqrt(hip_dx**2 + hip_dy**2)
    shoulder_speed = np.sqrt(shoulder_dx**2 + shoulder_dy**2)
    
    # Calculate vertical velocity (downward movement is positive)
    hip_vertical_velocity = hip_dy  # Positive = moving down
    shoulder_vertical_velocity = shoulder_dy
    
    return {
        'hip_speed': hip_speed,
        'shoulder_speed': shoulder_speed,
        'hip_vertical': hip_vertical_velocity,
        'shoulder_vertical': shoulder_vertical_velocity,
        'frames_analyzed': len(history)
    }

def detect_fall_by_velocity(history):
    """
    Detect falls based on movement velocity
    Returns: (is_fall_detected, fall_reason)
    """
    velocities = calculate_velocity(history)
    
    if not velocities or velocities['frames_analyzed'] < 5:
        return False, "Insufficient velocity data"
    
    # Define thresholds (these values may need to adjusted according to needed...)
    SPEED_THRESHOLD = 0.08  # Rapid movement threshold
    VERTICAL_THRESHOLD = 0.05  # Significant downward movement
    
    fall_detected = False
    reason = ""
    
    # Check for rapid overall movement
    avg_speed = (velocities['hip_speed'] + velocities['shoulder_speed']) / 2
    if avg_speed > SPEED_THRESHOLD:
        fall_detected = True
        reason = f"Rapid movement: {avg_speed:.3f}"
    
    # Check for significant downward movement (positive = down)
    avg_vertical = (velocities['hip_vertical'] + velocities['shoulder_vertical']) / 2
    if avg_vertical > VERTICAL_THRESHOLD:
        fall_detected = True
        reason = f"Downward movement: {avg_vertical:.3f}"
    
    return fall_detected, reason

def detect_fall_by_position(landmarks):
    """
    Detect falls based on body orientation (vertical vs horizontal)
    
    Key insight: 
    - Standing/sitting: Body is VERTICAL (shoulder-hip line is vertical)
    - Lying down: Body is HORIZONTAL (shoulder-hip line is horizontal)
    
    This works regardless of position in frame, camera angle, or body posture.
    """
    # Get shoulder and hip landmarks
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    
    # Calculate center points
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_center_x = (left_hip.x + right_hip.x) / 2
    hip_center_y = (left_hip.y + right_hip.y) / 2
    
    # Calculate the vertical span (how much vertical distance between shoulders and hips)
    vertical_distance = abs(shoulder_center_y - hip_center_y)
    
    # Calculate the horizontal span (how much horizontal distance between shoulders and hips)
    horizontal_distance = abs(shoulder_center_x - hip_center_x)
    
    # Calculate body orientation angle
    # When standing: vertical_distance is large, horizontal_distance is small
    # When lying down: vertical_distance is small, horizontal_distance is large
    
    # Avoid division by zero
    if vertical_distance < 0.001:
        vertical_distance = 0.001
    
    # Aspect ratio: horizontal / vertical
    # Standing: ratio ~0.0-0.3 (mostly vertical)
    # Lying down: ratio ~1.0+ (mostly horizontal)
    orientation_ratio = horizontal_distance / vertical_distance
    
    # Thresholds
    HORIZONTAL_THRESHOLD = 0.8  # If ratio > 0.6, body is too horizontal
    # VERTICAL_SPAN_THRESHOLD = 0.15  # If vertical span < 0.15, body is compressed/horizontal

    # --- NEW: Full-body bounding box aspect ratio ---
    all_x = [lm.x for lm in landmarks]
    all_y = [lm.y for lm in landmarks]
    bbox_width = max(all_x) - min(all_x)
    bbox_height = max(all_y) - min(all_y)

    if bbox_height < 0.001:
        bbox_height = 0.001

    bbox_ratio = bbox_width / bbox_height
    BBOX_THRESHOLD = 1.2  # Body wider than tall = likely fallen

    # --- Decision: either signal can trigger ---
    is_horizontal = orientation_ratio > HORIZONTAL_THRESHOLD
    is_wide_bbox = bbox_ratio > BBOX_THRESHOLD
    # body_is_low = hip_center_y > 0.60  # hips in lower portion of frame
    
    fall_detected = (is_horizontal or is_wide_bbox)
    reason = ""
    
    # Check 1: Body orientation (horizontal vs vertical)
    if fall_detected:
        reason = f"Horizontal body | Orientation: {orientation_ratio:.3f} | BBox: {bbox_ratio:.3f}"
    else:
        reason = f"Vertical body | Orientation: {orientation_ratio:.3f} | BBox: {bbox_ratio:.3f}"
    
    # Check 2: Vertical span too small (body lying flat)
    # if vertical_distance < VERTICAL_SPAN_THRESHOLD:
    #     fall_detected = True
    #     reason = f"Minimal vertical span: {vertical_distance:.3f}"
    
    return fall_detected, reason

def calculate_fall_confidence(is_fall_angle, is_fall_velocity, is_fall_position):
    """
    Calculate confidence score for fall detection using weighted algorithm
    
    Weighting:
    - Position: 50% (most definitive indicator)
    - Angle: 25% (can have false positives from bending)
    - Velocity: 25% (can have false positives from quick movements)
    
    Returns: (confidence_percentage, confidence_level, active_detectors)
    """
    # Define weights (must sum to 100)
    POSITION_WEIGHT = 50
    ANGLE_WEIGHT = 25
    VELOCITY_WEIGHT = 25
    
    # Calculate confidence score
    confidence = 0
    active_detectors = []
    
    if is_fall_position:
        confidence += POSITION_WEIGHT
        active_detectors.append("Position")
    
    if is_fall_angle:
        confidence += ANGLE_WEIGHT
        active_detectors.append("Angle")
    
    if is_fall_velocity:
        confidence += VELOCITY_WEIGHT
        active_detectors.append("Velocity")
    
    # Determine confidence level based on thresholds
    if confidence >= 61:
        confidence_level = "FALL_DETECTED"
    elif confidence >= 31:
        confidence_level = "SUSPICIOUS"
    else:
        confidence_level = "NORMAL"
    
    return confidence, confidence_level, active_detectors

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
        
# def display_positions_on_frame(frame, positions):
#     """Display vertical positions on the frame for debugging"""
#     if positions:
#         y_offset = 130  # Start below the angles display
#         cv2.putText(frame, f"Nose Y: {positions['nose_y']:.3f}", 
#                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
#         cv2.putText(frame, f"Hip Y: {positions['hip_y']:.3f}", 
#                    (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
#         cv2.putText(frame, f"Shoulder Y: {positions['shoulder_y']:.3f}", 
#                    (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
#         cv2.putText(frame, f"Knee Y: {positions['knee_y']:.3f}", 
#                    (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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

def display_fall_alert(frame, confidence, confidence_level, reason):
    """
    Display fall alert with confidence scoring
    - Green: Normal (0-30%)
    - Yellow: Suspicious (31-60%)
    - Red: Fall Detected (61-100%)
    """
    y_base = frame.shape[0] - 60
    
    if confidence_level == "FALL_DETECTED":
        # Red alert - Fall detected
        cv2.putText(frame, f"FALL DETECTED! (Confidence: {confidence}%)", 
                   (10, y_base),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(frame, reason, 
                   (10, y_base + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    elif confidence_level == "SUSPICIOUS":
        # Yellow warning - Suspicious activity
        cv2.putText(frame, f"SUSPICIOUS ACTIVITY (Confidence: {confidence}%)", 
                   (10, y_base),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
        cv2.putText(frame, reason, 
                   (10, y_base + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    else:
        # Green status - Normal
        cv2.putText(frame, f"Status: Normal (Confidence: {confidence}%)", 
                   (10, frame.shape[0] - 20),
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

    # Initialize landmark history for velocity tracking
    landmark_history = []
    MAX_HISTORY_FRAMES = 10

    # Time-window buffer for temporal smoothing
    confidence_window = []
    WINDOW_DURATION_SECONDS = 2
    FPS = 30  # match your video FPS
    # If we want to set FPS value dynamically 
    # (commented out because might not work in some cases where metadata is missing)
    # FPS = int(cap.get(cv2.CAP_PROP_FPS))
    WINDOW_SIZE = WINDOW_DURATION_SECONDS * FPS  # = 60 frames

    current_window_level = "NORMAL"
    consecutive_suspicious_count = 0
    last_notification_time = None
    last_suspicious_count_time = None
    FALL_NOTIFICATION_COOLDOWN = 120  # seconds
    SUSPICIOUS_NOTIFICATION_COOLDOWN = 60  # seconds

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

            # Add this in the main loop where you process pose results
            # positions = get_vertical_positions(pose_results.pose_landmarks.landmark)
            # display_positions_on_frame(processed_frame, positions)
            # print(f"Positions - Nose: {positions['nose_y']:.3f}, Hip: {positions['hip_y']:.3f}")

            # Test the position detection
            # is_fall_position, reason_position = detect_fall_by_position(positions)
            # print(f"Position Detection - Fall: {is_fall_position}, Reason: {reason_position}")

            update_landmark_history(pose_results.pose_landmarks.landmark, landmark_history, MAX_HISTORY_FRAMES)
            # velocities = calculate_velocity(landmark_history)
            # if velocities:
            #     print(f"Hip speed: {velocities['hip_speed']:.4f}, Vertical: {velocities['hip_vertical']:.4f}")
            
            # Detect fall and display alert/status
            # is_fall, reason = detect_fall_by_angles(angles)
            # display_fall_alert(processed_frame, is_fall, reason)

            # Detect fall using both algorithms
            is_fall_angle, reason_angle = detect_fall_by_angles(angles)
            is_fall_velocity, reason_velocity = detect_fall_by_velocity(landmark_history)
            is_fall_position, reason_position = detect_fall_by_position(pose_results.pose_landmarks.landmark)

            # Calculate confidence score
            confidence, confidence_level, active_detectors = calculate_fall_confidence(
                is_fall_angle, is_fall_velocity, is_fall_position
            )

            # Store confidence score in time-window buffer
            confidence_window.append(confidence)

            # Keep buffer trimmed to window size
            if len(confidence_window) > WINDOW_SIZE:
                confidence_window.pop(0)

            # Evaluate every frame once buffer is full
            if len(confidence_window) == WINDOW_SIZE:
                # Count frames where fall was detected in the last 2 seconds
                fall_frame_count = sum(1 for c in confidence_window if c >= 61)
                suspicious_frame_count = sum(1 for c in confidence_window if 31 <= c <= 60)
                
                # Calculate ratio of fall/suspicious frames in the window
                fall_ratio = fall_frame_count / len(confidence_window)
                suspicious_ratio = suspicious_frame_count / len(confidence_window)

                # Determine current window level
                if fall_ratio >= 0.5:
                    current_window_level = "FALL_DETECTED"
                elif suspicious_ratio >= 0.5:
                    current_window_level = "SUSPICIOUS"
                else:
                    current_window_level = "NORMAL"
                
                print(f"[Rolling Window] Fall ratio: {fall_ratio:.2f} | Suspicious ratio: {suspicious_ratio:.2f} → {current_window_level}")
                
                # Escalation logic
                current_time = time.time()

                if current_window_level == "FALL_DETECTED":
                    consecutive_suspicious_count = 0  # reset
                    # Check if fall notification cooldown has expired
                    if last_notification_time is None or (current_time - last_notification_time) >= FALL_NOTIFICATION_COOLDOWN:
                        last_notification_time = current_time
                        print("🚨 REAL FALL CONFIRMED - SEND FALL NOTIFICATION")
                
                elif current_window_level == "SUSPICIOUS":
                    if last_suspicious_count_time is None or (current_time - last_suspicious_count_time) >= WINDOW_DURATION_SECONDS:
                        last_suspicious_count_time = current_time
                        consecutive_suspicious_count += 1
                        print(f"Suspicious count: {consecutive_suspicious_count}/2")
                        if consecutive_suspicious_count >= 2:
                            consecutive_suspicious_count = 0  # reset after triggering
                            # Check if suspicious notification cooldown has expired
                            if last_notification_time is None or (current_time - last_notification_time) >= SUSPICIOUS_NOTIFICATION_COOLDOWN:
                                last_notification_time = current_time
                                print("⚠️ SUSTAINED SUSPICIOUS ACTIVITY - SEND WARNING NOTIFICATION")
                
                else:  # NORMAL
                    consecutive_suspicious_count = 0  # reset on normal

            # Collect all reasons from active detectors
            reasons = []
            if is_fall_angle:
                reasons.append(f"Angle: {reason_angle}")
            if is_fall_velocity:
                reasons.append(f"Velocity: {reason_velocity}")
            if is_fall_position:
                reasons.append(f"Position: {reason_position}")

            # Combine reasons into one string
            combined_reason = " | ".join(reasons) if reasons else "No anomalies detected"

            # Display result with confidence scoring
            display_fall_alert(processed_frame, confidence, confidence_level, combined_reason)

            # # Console output for monitoring
            # if confidence_level == "FALL_DETECTED":
            #     print(f"🚨 FALL DETECTED! - Confidence: {confidence}% - Detectors: {', '.join(active_detectors)}")
            #     print(f"   Reason: {combined_reason}")
            #     print(f"   >>> SEND CRITICAL ALERT NOTIFICATION <<<")
            # elif confidence_level == "SUSPICIOUS":
            #     print(f"⚠️  SUSPICIOUS ACTIVITY - Confidence: {confidence}% - Detectors: {', '.join(active_detectors)}")
            #     print(f"   Reason: {combined_reason}")
            #     print(f"   >>> SEND WARNING NOTIFICATION <<<")

        else:
            # If no pose detected, show normal status
            display_fall_alert(processed_frame, 0, "NORMAL", "No pose detected")

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