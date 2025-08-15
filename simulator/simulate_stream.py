import requests
import pandas as pd
import json
import time
from datetime import datetime
import os

CSV_FILE = "data/faulty_data.csv"
HTTP_ENDPOINT = "http://localhost:8000/api/v1/stream"
MACHINE_ID = "machine_001"
COMMAND_FILE = "fault_command.json"

# Dummy fault class registry for demonstration
FAULT_CLASSES = {
    "SpikeFault": lambda duration=10: f"SpikeFault(duration={duration})",
    "TempRampFault": lambda duration=10: f"TempRampFault(duration={duration})"
}

def run_simulation():
    df = pd.read_csv(CSV_FILE)
    active_fault = None
    fault_end_time = None

    while True:
        # Check for command file
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, "r") as f:
                    command = json.load(f)
                machine_id = command.get("machine_id")
                fault_type = command.get("fault_type")
                custom_duration = command.get("duration", 10)
                fault_cls = FAULT_CLASSES.get(fault_type)
                if fault_cls:
                    active_fault = fault_cls(duration=custom_duration)
                    fault_end_time = time.time() + custom_duration
                    print(f"Injecting fault '{fault_type}' into '{machine_id}' for {custom_duration} seconds.")
                else:
                    print(f"Unknown fault type: {fault_type}")
                # Remove the command file after processing
                os.remove(COMMAND_FILE)
            except Exception as e:
                print(f"Error processing command file: {e}")

        for _, row in df.iterrows():
            # Here you would apply the fault to the data if active_fault is set
            payload = {
                "machine_id": MACHINE_ID,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sensors": {
                    "vibration_x": row["vibration_x"],
                    "vibration_y": row["vibration_y"],
                    "temperature": row["temperature"],
                    "current": row["current"]
                }
            }
            # Optionally, modify payload based on active_fault here

            response = requests.post(HTTP_ENDPOINT, json=payload)
            print(f"Sent: {json.dumps(payload)}")
            print(f"Response status code: {response.status_code}")

            # Deactivate fault if duration has passed
            if active_fault and fault_end_time and time.time() >= fault_end_time:
                print(f"Fault duration ended. Clearing active fault.")
                active_fault = None
                fault_end_time = None

            time.sleep(1)

if __name__ == "__main__":
    run_simulation()
