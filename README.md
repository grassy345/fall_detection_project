# Elderly Fall Detection System

A computer vision-based fall detection system using MediaPipe and OpenCV for real-time monitoring of elderly individuals. This system uses pose estimation to detect human body landmarks and analyze movement patterns to identify potential falls.

## Features

- Real-time pose detection using MediaPipe
- Webcam integration for live monitoring
- Skeleton visualization overlay
- Cross-platform support (Windows, Linux, macOS)
- Optimized for single-person detection

## Prerequisites

- Python 3.7 or higher
- Webcam or USB camera
- For WSL users: USB camera forwarding setup (see setup instructions)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fall-detection-system.git
cd fall-detection-system
```

### 2. Create Virtual Environment

```bash
python3 -m venv fall_detection_env
source fall_detection_env/bin/activate  # On Windows: fall_detection_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Test Camera Setup

First, test if your camera is working:

```bash
python test_camera.py
```

You should see a camera feed window. If you encounter issues, see the troubleshooting section.

### 5. Run the Fall Detection System

```bash
python fall_detection.py
```

Press 'q' to quit the application.

## WSL Setup (Windows Users)

If you're using WSL2 on Windows, you'll need to forward your USB camera:

1. Install usbipd on Windows:
   ```powershell
   winget install usbipd
   ```

2. List and attach your camera:
   ```powershell
   usbipd list
   usbipd attach --wsl --busid <your-camera-busid>
   ```

3. In WSL, install camera utilities:
   ```bash
   sudo apt update
   sudo apt install v4l-utils
   ```

## Project Structure

```
fall-detection-system/
├── fall_detection.py      # Main fall detection application
├── test_camera.py         # Camera testing utility
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore           # Git ignore rules
```

## How It Works

1. **Camera Input**: Captures live video feed from webcam
2. **Pose Detection**: Uses MediaPipe to detect 33 body landmarks
3. **Skeleton Visualization**: Overlays detected pose on video feed
4. **Fall Analysis**: (In development) Analyzes pose data to detect falls

## Current Status

✅ Camera integration and pose detection  
✅ Real-time skeleton visualization  
🚧 Fall detection algorithm (in progress)  
🚧 Alert system (planned)  
🚧 Multi-person detection (planned)  

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Test your changes with `test_camera.py` before committing
- Update README.md if you add new features

## Troubleshooting

### Camera Not Working
- Check if camera is being used by another application
- Try different camera indices (0, 1, 2) in `test_camera.py`
- For WSL users, ensure USB forwarding is properly configured

### Performance Issues
- Lower the camera resolution in `setup_camera()` function
- Reduce FPS if needed
- Close other applications using the camera

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## Future Enhancements

- [ ] Implement fall detection algorithm
- [ ] Add email/SMS alert system
- [ ] Support for multiple camera angles
- [ ] Machine learning model for improved accuracy
- [ ] Mobile app integration
- [ ] Cloud-based monitoring dashboard

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- MediaPipe team for the pose estimation framework
- OpenCV community for computer vision tools
- Contributors and testers

## Contact

For questions or suggestions, please open an issue on GitHub or contact [your-email@example.com].

---

**Note**: This is a college project focused on learning computer vision concepts. The system is designed for educational purposes and should not be used as the sole monitoring solution for elderly care without proper testing and validation.
