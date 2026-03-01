import time
import firebase_admin
from firebase_admin import credentials, db

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_KEY = "serviceAccountKey.json"
DATABASE_URL = "https://fall-detection-alarm-system-default-rtdb.asia-southeast1.firebasedatabase.app/"
STATUS_FILE = "status.txt"
POLL_INTERVAL = 1        # seconds between status.txt reads
ACK_POLL_INTERVAL = 2    # seconds between Firebase acknowledged checks
# ──────────────────────────────────────────────────────────────────────────────


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


def send_to_firebase(status):
    """Write fall_status, timestamp, and acknowledged to /fall_alert atomically."""
    ref = db.reference("fall_alert")
    timestamp = time.strftime("%d-%m-%Y %H:%M:%S")
    ref.update({
        "fall_status": status,
        "timestamp": timestamp,
        "acknowledged": False
    })
    print(f"[Firebase] Sent → fall_status: {status}, timestamp: {timestamp}, acknowledged: False")


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
                print("[INFO] Now waiting for caregiver acknowledgement...")

            elif current_status == "NORMAL":
                # Instance 1: mirror NORMAL back to Firebase so server stays in sync
                send_to_firebase("NORMAL")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()