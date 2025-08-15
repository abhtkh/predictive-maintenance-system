import argparse
import asyncio
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type

import httpx
import numpy as np
import pandas as pd

# --- Configuration ---
COMMAND_FILE = "fault_command.json"
GROUND_TRUTH_LOG = "ground_truth.log"

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Fault Models ---
class FaultBase:
    """Base class for event-based, temporary faults."""
    def __init__(self, duration: float = 30.0):
        self.end_time = time.time() + duration

    def apply(self, sensors: Dict[str, float], machine: "Machine") -> Dict[str, float]:
        return sensors

class SpikeFault(FaultBase):
    """Represents intermittent mechanical shocks, primarily affecting vibration."""
    def apply(self, sensors, machine):
        if random.random() < 0.3:  # Spikes are not continuous
            sensors["vibration_x"] += 0.15 + random.random() * 0.1
            sensors["vibration_y"] += 0.10 + random.random() * 0.08
        return sensors

class TempRampFault(FaultBase):
    """Represents a cooling system issue, causing a slow rise in temperature."""
    def apply(self, sensors, machine):
        elapsed = self.end_time - time.time()
        ramp_factor = 1.0 - (elapsed / 30.0) # Scale from 0 to 1 over the duration
        sensors["temperature"] += 15.0 * ramp_factor
        return sensors

class OverloadFault(FaultBase):
    """Represents the machine being pushed beyond its limits, primarily affecting current."""
    def apply(self, sensors, machine):
        sensors["current"] += 5.0 + random.random() * 1.5
        sensors["temperature"] += 2.0 # Overload also generates some heat
        return sensors

FAULT_MAP: Dict[str, Type[FaultBase]] = {
    "SpikeFault": SpikeFault,
    "TempRampFault": TempRampFault,
    "OverloadFault": OverloadFault,
}

# --- High-Fidelity Machine Model ---
@dataclass
class Machine:
    machine_id: str
    base_rpm: float = 1800.0
    degradation_rate: float = 0.00001 # VERY slow health decrease per second
    t: float = 0.0
    health: float = 1.0
    active_fault: Optional[FaultBase] = None

    def step(self, dt: float = 0.1) -> None:
        """Advance the machine's state by one time step."""
        self.t += dt
        # Apply continuous, slow degradation
        self.health -= self.degradation_rate * dt
        self.health = max(0.0, self.health)

    def read_sensors(self) -> Dict[str, float]:
        """Generate sensor values based on current health and any active fault."""
        freq = self.base_rpm / 60.0
        # Baseline noise and readings are now influenced by overall health
        health_factor = 1.0 - self.health
        vib_amp = 0.003 + health_factor * 0.01
        temp_base = 45.0 + health_factor * 20.0
        current_base = 6.0 + health_factor * 2.0

        sensors = {
            "vibration_x": vib_amp * math.sin(2 * math.pi * freq * self.t) + np.random.normal(0, vib_amp * 0.5),
            "vibration_y": vib_amp * 0.8 * math.sin(2 * math.pi * freq * self.t + 0.2) + np.random.normal(0, vib_amp * 0.5),
            "temperature": temp_base + np.random.normal(0, 0.2),
            "current": current_base + np.random.normal(0, 0.1),
        }

        # Apply the temporary, acute fault if one is active
        if self.active_fault:
            if time.time() > self.active_fault.end_time:
                log_ground_truth(self.machine_id, self.active_fault.__class__.__name__, "end")
                self.active_fault = None
            else:
                sensors = self.active_fault.apply(sensors, self)

        return {k: float(v) for k, v in sensors.items()}

    def trigger_fault_event(self, fault_cls: Type[FaultBase], duration: float) -> None:
        """Triggers a new temporary fault event."""
        if self.active_fault:
            logger.warning(f"Machine {self.machine_id} already has an active fault. Overwriting.")
        self.active_fault = fault_cls(duration=duration)
        log_ground_truth(self.machine_id, fault_cls.__class__.__name__, "start")

def log_ground_truth(machine_id: str, fault_type: str, status: str) -> None:
    # ... (This function is unchanged)
    ts = datetime.now(timezone.utc).isoformat()
    with open(GROUND_TRUTH_LOG, "a") as f:
        f.write(f"{ts},{machine_id},{fault_type},{status}\n")


# --- Main Simulation & Control Logic ---
async def stream_http(session: httpx.AsyncClient, url: str, payload: dict) -> None:
    # ... (This function is unchanged)
    try:
        r = await session.post(url, json=payload, timeout=5.0)
        if r.is_error:
            logger.warning(f"HTTP {r.status_code}: {r.text} for {payload['machine_id']}")
    except Exception as e:
        logger.error(f"HTTP send failed: {e}")

def check_commands(machines: List[Machine]) -> None:
    # ... (This function is now simpler)
    if not os.path.exists(COMMAND_FILE):
        return
    try:
        with open(COMMAND_FILE) as f:
            cmd = json.load(f)
        
        target_machine = next((m for m in machines if m.machine_id == cmd.get("machine_id")), None)
        fault_cls = FAULT_MAP.get(cmd.get("fault_type"))
        
        if target_machine and fault_cls:
            duration = float(cmd.get("duration", 30))
            target_machine.trigger_fault_event(fault_cls, duration)
            logger.info(f"Injected {cmd['fault_type']} into {cmd['machine_id']} for {duration}s")
    except Exception as e:
        logger.error(f"Command processing failed: {e}")
    finally:
        os.remove(COMMAND_FILE)

async def run_simulation(args):
    # ... (This function is now simpler and cleaner)
    if os.path.exists(COMMAND_FILE): os.remove(COMMAND_FILE)
    if os.path.exists(GROUND_TRUTH_LOG): os.remove(GROUND_TRUTH_LOG)
    log_ground_truth("system", "simulation", "start")

    machines = [Machine(f"M{i+1}") for i in range(args.machines)]
    async with httpx.AsyncClient() as session:
        while True:
            check_commands(machines)
            payloads = []
            for m in machines:
                m.step()
                # Payload structure is unchanged
                payload = {
                    "machine_id": m.machine_id,
                    "sensors": m.read_sensors() # The read_sensors method is now much smarter
                }
                payloads.append(payload)

            tasks = [stream_http(session, args.url, p) for p in payloads]
            await asyncio.gather(*tasks)
            await asyncio.sleep(args.rate / 1000.0)

def parse_args():
    # ... (This function is unchanged)
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/api/v1/stream")
    p.add_argument("--machines", type=int, default=3)
    p.add_argument("--rate", type=int, default=1000)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_simulation(args))
    except KeyboardInterrupt:
        logger.info("Simulator stopped by user.")