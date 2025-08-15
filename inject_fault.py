import json
import argparse

COMMAND_FILE = "fault_command.json"

def main():
    fault_choices = ["SpikeFault", "TempRampFault", "OverloadFault"]

    parser = argparse.ArgumentParser(
        description="Inject a temporary fault event into a running simulator.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--machine-id", required=True, help="ID of the machine to affect (e.g., M1).")
    parser.add_argument("--fault-type", required=True, choices=fault_choices, help="Type of fault to inject.")
    parser.add_argument("--duration", type=int, default=45, help="Duration of the fault in seconds (default: 45).")

    args = parser.parse_args()

    command = {
        "machine_id": args.machine_id,
        "fault_type": args.fault_type,
        "duration": args.duration,
    }

    try:
        with open(COMMAND_FILE, "w") as f:
            json.dump(command, f)
        print(f"\n✅ Command sent: Injecting '{args.fault_type}' into '{args.machine_id}' for {args.duration}s.\n")
    except Exception as e:
        print(f"\n❌ Error creating command file: {e}\n")

if __name__ == "__main__":
    main()