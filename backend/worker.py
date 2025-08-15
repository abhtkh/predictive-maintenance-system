import sys
import os
import redis
import json
import time
from typing import Dict, List

# --- Path Fix for Direct Execution ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Standard Library Imports ---
import smtplib
from email.mime.text import MIMEText

# --- Third-Party Imports ---
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from sqlalchemy.orm import Session

# --- Local Application Imports ---
from backend.inference import InferenceWrapper
from backend.database import SessionLocal
from backend import crud, models
from backend.config import (
    REDIS_HOST, REDIS_PORT, INGEST_STREAM_NAME, RESULT_CHANNEL_NAME,
    CONSUMER_GROUP_NAME, ALERT_THRESHOLD, ALERT_DURATION_SECONDS,
    SMTP_ENABLED, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    SMTP_SENDER_EMAIL
)

WORKER_NAME = f"pdm_worker_{os.getpid()}"

# --- Prometheus Metrics (Unchanged) ---
MESSAGES_PROCESSED = Counter("pdm_messages_processed_total", "...")
LAST_RECONSTRUCTION_ERROR = Gauge("pdm_last_reconstruction_error", "...", ["machine_id"])
INFERENCE_LATENCY = Histogram("pdm_inference_latency_seconds", "...")
PER_SENSOR_ERROR = Gauge("pdm_per_sensor_error", "...", ["machine_id", "sensor_name"])

# --- Stateful Machine Tracker (with Debugging) ---
class MachineState:
    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self.in_alert_state = False
        self.alert_start_time = None
        self.notification_sent = False

    def update(self, anomaly_score: float, db: Session):
        is_above_threshold = anomaly_score >= ALERT_THRESHOLD

        # --- DEBUG LOG #1: Log every single update ---
        print(f"[DEBUG {self.machine_id}] Score: {anomaly_score:.2f} | Threshold: {ALERT_THRESHOLD} | Is Above: {is_above_threshold} | In Alert State: {self.in_alert_state}")

        if is_above_threshold and not self.in_alert_state:
            self.in_alert_state = True
            self.alert_start_time = time.time()
            self.notification_sent = False
            print(f"ALERT START for {self.machine_id}: Score {anomaly_score:.2f} crossed threshold {ALERT_THRESHOLD}")

        elif not is_above_threshold and self.in_alert_state:
            if self.notification_sent:
                duration = time.time() - self.alert_start_time
                send_email_notifications(db, self.machine_id, anomaly_score, duration, is_resolved=True)
            self.in_alert_state = False
            self.alert_start_time = None
            print(f"ALERT END for {self.machine_id}: Score {anomaly_score:.2f} is back to normal.")
        
        if self.in_alert_state and not self.notification_sent:
            duration = time.time() - self.alert_start_time

            # --- DEBUG LOG #2: Log the duration check ---
            print(f"[DEBUG {self.machine_id}] Checking duration... Current: {duration:.1f}s | Required: {ALERT_DURATION_SECONDS}s")
            
            if duration >= ALERT_DURATION_SECONDS:
                print(f"SUSTAINED ALERT for {self.machine_id}: Sending notification.") # This is the line we're looking for
                send_email_notifications(db, self.machine_id, anomaly_score, duration, is_resolved=False)
                self.notification_sent = True

# --- Email Function (Unchanged) ---
def send_email_notifications(db: Session, machine_id: str, score: float, duration: float, is_resolved: bool):
    # ... (This function remains exactly the same as before)
    if not SMTP_ENABLED:
        print("SMTP is disabled in config. Skipping email notifications.")
        return
    recipients = crud.get_alert_recipients(db)
    if not recipients:
        print(f"No recipients configured for email alerts. Skipping notification for {machine_id}.")
        return
    if is_resolved:
        subject = f"✅ RESOLVED: PdM Alert for Machine {machine_id}"
        body_template = ("Dear {user_name},\n\nThe alert for machine {machine_id} has been resolved.\n\n"
                         "The anomaly score is now {score:.2f}, which is below the threshold of {threshold}.\n"
                         "The alert was active for approximately {duration:.1f} minutes.\n\n"
                         "The system is now considered back to normal operation.")
    else:
        subject = f"🚨 CRITICAL: PdM Alert for Machine {machine_id}"
        body_template = ("Dear {user_name},\n\nA critical alert has been triggered for machine {machine_id}.\n\n"
                         "  - Current Anomaly Score: {score:.2f}\n"
                         "  - Alert Threshold: {threshold}\n"
                         "  - Sustained Duration: >{duration:.1f} minutes.\n\n"
                         "Please investigate the machine's condition immediately.")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            for user in recipients:
                body = body_template.format(user_name=user.full_name or user.username, machine_id=machine_id, score=score, threshold=ALERT_THRESHOLD, duration=duration / 60)
                msg = MIMEText(body)
                msg['Subject'] = subject
                msg['From'] = SMTP_SENDER_EMAIL
                msg['To'] = user.email
                server.send_message(msg)
                print(f"Email sent successfully to {user.email} for alert on {machine_id}")
    except Exception as e:
        print(f"Error: Failed to send email notifications. Details: {e}")


# --- Main Worker Logic (Unchanged) ---
def main():
    print("Starting Predictive Maintenance Worker (with DB-Driven Alerting)...")
    start_http_server(8001)
    db = SessionLocal()
    pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client = redis.Redis(connection_pool=pool)
    inference_wrapper = InferenceWrapper()
    machine_states: Dict[str, MachineState] = {}
    try:
        redis_client.xgroup_create(INGEST_STREAM_NAME, CONSUMER_GROUP_NAME, id="0", mkstream=True)
    except redis.exceptions.ResponseError: pass
    while True:
        try:
            resp = redis_client.xreadgroup(groupname=CONSUMER_GROUP_NAME, consumername=WORKER_NAME, streams={INGEST_STREAM_NAME: '>'}, count=1, block=5000)
            if not resp: continue
            for _, messages in resp:
                for msg_id, msg_data in messages:
                    try:
                        payload = json.loads(msg_data['data'])
                        machine_id = payload["machine_id"]
                        if machine_id not in machine_states:
                            machine_states[machine_id] = MachineState(machine_id)
                        state = machine_states[machine_id]
                        sensor_keys = ["vibration_x", "vibration_y", "temperature", "current"]
                        sensor_values = [payload["sensors"].get(k, 0.0) for k in sensor_keys]
                        prediction = inference_wrapper.predict(sensor_values)
                        anomaly_score = prediction.get("total_reconstruction_error", 0.0)
                        
                        MESSAGES_PROCESSED.inc()
                        LAST_RECONSTRUCTION_ERROR.labels(machine_id=machine_id).set(anomaly_score)
                        per_sensor_errors = prediction.get("per_sensor_error", [])
                        if len(per_sensor_errors) == len(sensor_keys):
                            for i, key in enumerate(sensor_keys):
                                PER_SENSOR_ERROR.labels(machine_id=machine_id, sensor_name=key).set(per_sensor_errors[i])
                        
                        state.update(anomaly_score, db)
                        
                        result = {"machine_id": machine_id, "timestamp": time.time(), "alert_active": state.in_alert_state, **prediction}
                        redis_client.publish(RESULT_CHANNEL_NAME, json.dumps(result))
                        redis_client.xack(INGEST_STREAM_NAME, CONSUMER_GROUP_NAME, msg_id)
                    except Exception as msg_exc:
                        print(f"Error processing message {msg_id}: {msg_exc}")
        except Exception as e:
            print(f"Worker error: {e}")
            db.close()
            time.sleep(2)
            db = SessionLocal()

if __name__ == "__main__":
    main()