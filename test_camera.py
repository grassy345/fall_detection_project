import cv2

def test_camera_with_format():
    cap = cv2.VideoCapture(0)
    
    # Set specific format and resolution
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Alternative: try YUYV format
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
    
    if cap.isOpened():
        print("Camera opened successfully")
        ret, frame = cap.read()
        if ret:
            print(f"✓ Frame captured! Shape: {frame.shape}")
            cv2.imshow('Camera Test', frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()
            return True
        else:
            print("✗ Could not read frame")
    else:
        print("✗ Could not open camera")
    
    cap.release()
    return False

# Test it
test_camera_with_format()