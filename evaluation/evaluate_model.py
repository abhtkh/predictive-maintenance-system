import sys
import os
import redis
import json
import time
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Add project root to sys.path to resolve imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.config import REDIS_HOST, REDIS_PORT, RESULT_CHANNEL_NAME

# --- Configuration ---
PREDICTIONS_LOG = "predictions.log"
GROUND_TRUTH_LOG = "ground_truth.log"
ALERTING_THRESHOLD = 5000.0

def analyze_results():
    """
    Reads prediction and ground truth logs, merges them, and calculates
    performance metrics.
    """
    # This block is now correctly indented
    print("\n--- Analyzing Results ---")
    try:
        # 1. Load predictions
        preds_df = pd.read_json(PREDICTIONS_LOG, lines=True)
        preds_df['timestamp'] = pd.to_datetime(preds_df['timestamp'])
        preds_df = preds_df.sort_values('timestamp').set_index('timestamp')
        print(f"Loaded {len(preds_df)} predictions.")

        # 2. Load ground truth
        truth_df = pd.read_csv(
            GROUND_TRUTH_LOG,
            names=['timestamp', 'machine_id', 'fault_type', 'status']
        )
        truth_df['timestamp'] = pd.to_datetime(truth_df['timestamp'])
        truth_df = truth_df.sort_values('timestamp')
        print(f"Loaded {len(truth_df)} ground truth events.")

        # 3. Create a unified timeline with a boolean `is_fault` state
        # Create a helper function to determine the fault state over time
        def get_fault_intervals(df):
            intervals = []
            for machine, group in df.groupby('machine_id'):
                starts = group[group['status'] == 'start']['timestamp']
                ends = group[group['status'] == 'end']['timestamp']
                for start in starts:
                    # Find the corresponding end
                    end = ends[ends > start].min()
                    if pd.notna(end):
                        intervals.append({'machine_id': machine, 'start': start, 'end': end})
            return pd.DataFrame(intervals)

        fault_intervals = get_fault_intervals(truth_df)

        # 4. Label the predictions based on the fault intervals
        preds_df['is_fault'] = False
        for _, row in fault_intervals.iterrows():
            mask = (preds_df['machine_id'] == row['machine_id']) & \
                   (preds_df.index >= row['start']) & \
                   (preds_df.index <= row['end'])
            preds_df.loc[mask, 'is_fault'] = True

        # --- 5. Calculate Metrics ---
        y_true = preds_df['is_fault']
        y_scores = preds_df['reconstruction_error']
        y_pred_at_threshold = (y_scores > ALERTING_THRESHOLD)

        precision = precision_score(y_true, y_pred_at_threshold, zero_division=0)
        recall = recall_score(y_true, y_pred_at_threshold, zero_division=0)
        f1 = f1_score(y_true, y_pred_at_threshold, zero_division=0)

        if len(y_true.unique()) > 1:
            auc = roc_auc_score(y_true, y_scores)
        else:
            auc = float('nan')

        print(f"\nResults for Threshold = {ALERTING_THRESHOLD}:")
        print(f"  Precision: {precision:.4f} (Of all alerts fired, this fraction were correct)")
        print(f"  Recall:    {recall:.4f} (Of all actual faults, this fraction were detected)")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\nThreshold-Independent Metric:")
        print(f"  ROC AUC Score: {auc:.4f} (How well the model separates normal vs. fault, 1.0 is perfect)")

    except FileNotFoundError as e:
        print(f"Error: Log file not found - {e}. Ensure simulator and harness ran correctly.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

def main():
    """
    Connects to Redis, listens for prediction results, logs them,
    and triggers analysis on exit.
    """
    if os.path.exists(PREDICTIONS_LOG):
        os.remove(PREDICTIONS_LOG)
        
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe(RESULT_CHANNEL_NAME)
    
    print(f"Listening for predictions on '{RESULT_CHANNEL_NAME}'. Press Ctrl+C to stop and analyze.")
    
    with open(PREDICTIONS_LOG, "a") as f:
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    data_str = message['data']
                    f.write(data_str + '\n')
                    print('.', end='', flush=True)
        except KeyboardInterrupt:
            print("\nStopping listener...")
        finally:
            analyze_results()

if __name__ == "__main__":
    main()