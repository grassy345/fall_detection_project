import os
import subprocess
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import time
import firebase_admin
from firebase_admin import credentials, db

# ─── CLOUDINARY CONFIGURATION ─────────────────────────────────────────────────
load_dotenv()
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY    = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
CLOUDINARY_PRESET     = "fall_detection_preset"
CLIP_FILE             = "fall_clip.mp4"

cloudinary.config(
    cloud_name = CLOUDINARY_CLOUD_NAME,
    api_key    = CLOUDINARY_API_KEY,
    api_secret = CLOUDINARY_API_SECRET
)
# ──────────────────────────────────────────────────────────────────────────────

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_KEY = "serviceAccountKey.json"
DATABASE_URL = "https://fall-detection-alarm-system-default-rtdb.asia-southeast1.firebasedatabase.app/"
STATUS_FILE = "status.txt"
POLL_INTERVAL = 1        # seconds between status.txt reads
ACK_POLL_INTERVAL = 2    # seconds between Firebase acknowledged checks
# ──────────────────────────────────────────────────────────────────────────────

def wait_for_clip_ready(filepath, triggered_at, timeout=60):
    """Poll file size until stable AND file was created after the alert triggered."""
    print(f"[Clip] Waiting for {filepath} to finish writing...")
    deadline = time.time() + timeout
    prev_size = -1

    while time.time() < deadline:
        if os.path.exists(filepath):
            file_modified_at = os.path.getmtime(filepath)
            if file_modified_at < triggered_at:
                # This is a stale file from before the alert — ignore it
                time.sleep(1)
                continue
            current_size = os.path.getsize(filepath)
            if current_size > 0 and current_size == prev_size:
                print(f"[Clip] File ready! Size: {current_size} bytes")
                return True
            prev_size = current_size
        time.sleep(1)

    print("[Clip] Timed out waiting for clip file.")
    return False


def upload_clip_to_cloudinary(filepath):
    """Re-encode to H.264 then upload to Cloudinary and return the public URL."""
    encoded_path = filepath.replace(".mp4", "_h264.mp4")

    # Re-encode to H.264 for universal browser and Android playback
    try:
        print("[ffmpeg] Re-encoding clip to H.264...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", filepath,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            encoded_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[ffmpeg] Re-encoding successful → {encoded_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ffmpeg] Re-encoding failed: {e}. Uploading raw file instead.")
        encoded_path = filepath      # fallback to raw if ffmpeg fails

    # Upload to Cloudinary
    try:
        print("[Cloudinary] Uploading clip...")
        result = cloudinary.uploader.upload(
            encoded_path,
            resource_type = "video",
            public_id     = "fall_clips/fall_clip",
            overwrite     = True,
            preset        = CLOUDINARY_PRESET
        )
        url = result.get("secure_url")
        print(f"[Cloudinary] Upload successful! URL: {url}")

        # Clean up encoded file if it was a separate file
        if encoded_path != filepath and os.path.exists(encoded_path):
            os.remove(encoded_path)

        return url
    except Exception as e:
        print(f"[Cloudinary] Upload failed: {e}")
        return None

def initialize_firebase():
    """Initialize Firebase connection using service account key."""
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
    firebase_admin.initialize_app(cred, {
        "databaseURL": DATABASE_URL
    })
    print("[OK] Firebase initialized successfully.")


def read_status_file():
    """Read status.txt. Creates it with NORMAL if it doesn't exist."""
    try:
        with open(STATUS_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("[WARN] status.txt not found. Creating it with NORMAL.")
        write_status_file("NORMAL")
        return "NORMAL"


def write_status_file(status):
    """Overwrite status.txt with the given status string."""
    with open(STATUS_FILE, "w") as f:
        f.write(status)
    print(f"[status.txt] Written: {status}")


def send_to_firebase(status, clip_url=None):
    """Write fall_status, timestamp, acknowledged and optionally clip_url atomically."""
    ref = db.reference("fall_alert")
    timestamp = time.strftime("%d-%m-%Y %H:%M:%S")
    payload = {
        "fall_status": status,
        "timestamp": timestamp,
        "acknowledged": False,
        "clip_url": clip_url if clip_url else ""
    }
    ref.update(payload)
    print(f"[Firebase] Sent → fall_status: {status}, timestamp: {timestamp}, clip_url: {clip_url or 'none'}")


def poll_for_acknowledgement():
    """
    Poll Firebase /fall_alert/acknowledged every ACK_POLL_INTERVAL seconds.
    When it turns True:
      - Write ACKNOWLEDGED to status.txt
      - Reset acknowledged to False AND fall_status to NORMAL in Firebase
    Returns True if acknowledged, False otherwise.
    """
    ref = db.reference("fall_alert/acknowledged")
    ack_value = ref.get()
    print(f"[Firebase] Polling acknowledged... current value: {ack_value}")

    if ack_value is True:
        print("[Firebase] Acknowledgement received from caregiver!")
        write_status_file("ACKNOWLEDGED")

        # Instance 2: reset server to clean NORMAL state atomically
        db.reference("fall_alert").update({
            "acknowledged": False,
            "fall_status": "NORMAL"
        })
        print("[Firebase] Reset → fall_status: NORMAL, acknowledged: False in Firebase.")
        return True

    return False


def main():
    # Clean up stale clip from previous session
    if os.path.exists(CLIP_FILE):
        os.remove(CLIP_FILE)
        print("[Startup] Removed stale fall_clip.mp4 from previous session.")
    
    initialize_firebase()
    print("[OK] Starting main polling loop. Press Ctrl+C to stop.\n")

    previous_status = None
    waiting_for_ack = False

    while True:
        # ── If waiting for caregiver acknowledgement, poll Firebase ──────────
        if waiting_for_ack:
            acknowledged = poll_for_acknowledgement()
            if acknowledged:
                waiting_for_ack = False
                previous_status = "ACKNOWLEDGED"  # prevent re-triggering on next read
            time.sleep(ACK_POLL_INTERVAL)
            continue  # skip status.txt polling until ack is received

        # ── Normal mode: poll status.txt for changes ─────────────────────────
        current_status = read_status_file()

        if current_status != previous_status:
            print(f"[CHANGE DETECTED] {previous_status} → {current_status}")
            previous_status = current_status

            if current_status in ("FALL_DETECTED", "SUSPICIOUS"):
                send_to_firebase(current_status)
                waiting_for_ack = True
                triggered_at = time.time()     # ← capture when alert fired
                print("[INFO] Now waiting for caregiver acknowledgement...")

                if wait_for_clip_ready(CLIP_FILE, triggered_at):
                    clip_url = upload_clip_to_cloudinary(CLIP_FILE)
                    if clip_url:
                        db.reference("fall_alert").update({"clip_url": clip_url})
                        print("[Firebase] clip_url updated in /fall_alert.")
                else:
                    print("[WARN] Clip not ready in time — alert sent without video.")

            elif current_status == "NORMAL":
                # Instance 1: mirror NORMAL back to Firebase so server stays in sync
                send_to_firebase("NORMAL")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()