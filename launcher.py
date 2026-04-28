import subprocess
import sys
import os
import signal
import time

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Paths to each environment's Python interpreter
FALL_DETECTION_PYTHON = "fall_detection_env/bin/python"
FIREBASE_PYTHON       = "firebase_admin_venv/bin/python"

# Scripts to run
FALL_DETECTION_SCRIPT = "fall_detection.py"
FIREBASE_SCRIPT       = "firebase_sender.py"
# ──────────────────────────────────────────────────────────────────────────────


def check_interpreters():
    """Verify both Python interpreters exist before launching."""
    missing = []
    if not os.path.exists(FALL_DETECTION_PYTHON):
        missing.append(FALL_DETECTION_PYTHON)
    if not os.path.exists(FIREBASE_PYTHON):
        missing.append(FIREBASE_PYTHON)
    if missing:
        print("[ERROR] Could not find the following Python interpreters:")
        for m in missing:
            print(f"  - {m}")
        print("\nMake sure you are running launcher.py from the project root directory.")
        sys.exit(1)


def launch():
    check_interpreters()

    print("[Launcher] Starting Fall Detection System...\n")

    # Launch both scripts as separate subprocesses with their own interpreters
    fall_proc = subprocess.Popen(
        [FALL_DETECTION_PYTHON, FALL_DETECTION_SCRIPT],
        # No stdout/stderr redirect — let both scripts print directly to this terminal
    )
    print(f"[Launcher] fall_detection.py started (PID: {fall_proc.pid})")

    # Small delay so fall_detection.py can initialize status.txt first
    time.sleep(2)

    firebase_proc = subprocess.Popen(
        [FIREBASE_PYTHON, FIREBASE_SCRIPT],
    )
    print(f"[Launcher] firebase_sender.py started (PID: {firebase_proc.pid})")
    print("\n[Launcher] Both processes running. Press Ctrl+C to stop both.\n")

    def shutdown(signum, frame):
        """Gracefully terminate both child processes on Ctrl+C."""
        print("\n[Launcher] Shutting down...")
        fall_proc.terminate()
        firebase_proc.terminate()

        # Give them a moment to exit cleanly
        try:
            fall_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            fall_proc.kill()

        try:
            firebase_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            firebase_proc.kill()

        print("[Launcher] Both processes terminated. Goodbye!")
        sys.exit(0)

    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Monitor both processes — if either crashes, shut down the other too
    while True:
        fall_exit = fall_proc.poll()
        firebase_exit = firebase_proc.poll()

        if fall_exit is not None:
            print(f"\n[Launcher] fall_detection.py exited unexpectedly (code: {fall_exit}). Stopping firebase_sender.py too.")
            firebase_proc.terminate()
            sys.exit(1)

        if firebase_exit is not None:
            print(f"\n[Launcher] firebase_sender.py exited unexpectedly (code: {firebase_exit}). Stopping fall_detection.py too.")
            fall_proc.terminate()
            sys.exit(1)

        time.sleep(2)


if __name__ == "__main__":
    launch()