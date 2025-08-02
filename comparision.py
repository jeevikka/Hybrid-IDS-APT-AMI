import pandas as pd
import re
from datetime import datetime

# File paths
predictions_file = "C:\Users\prati\OneDrive\Desktop\majorproject\real traffic\predictions.txt"
snortlog_file = "C:\Users\prati\OneDrive\Desktop\majorproject\real traffic\snortlog.txt"

# Read predictions.txt
predictions = []
with open(predictions_file, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 4:
            timestamp, src_ip, dest_ip, prediction = parts
            predictions.append((timestamp, src_ip, dest_ip, int(prediction)))

# Convert predictions to DataFrame
pred_df = pd.DataFrame(predictions, columns=["timestamp", "src_ip", "dest_ip", "prediction"])
pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])

# Read snortlog.txt
snort_alerts = []
with open(snortlog_file, 'r') as f:
    for line in f:
        match = re.search(r'(\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)\s+.*?\{TCP\}\s+(\d+\.\d+\.\d+\.\d+):\d+\s+->\s+(\d+\.\d+\.\d+\.\d+):\d+', line)
        if match:
            timestamp_str, src_ip, dest_ip = match.groups()
            timestamp = datetime.strptime(timestamp_str, "%m/%d-%H:%M:%S.%f")
            timestamp = timestamp.replace(year=datetime.now().year)  # Ensure full datetime format
            snort_alerts.append((timestamp, src_ip, dest_ip, 1))  # 1 = Detected alert

# Convert snort logs to DataFrame
snort_df = pd.DataFrame(snort_alerts, columns=["timestamp", "src_ip", "dest_ip", "snort_detected"])

# Round timestamps to seconds for better matching
pred_df["timestamp"] = pred_df["timestamp"].dt.round("S")
snort_df["timestamp"] = snort_df["timestamp"].dt.round("S")

# Merge both DataFrames on timestamp, source, and destination IP
merged_df = pred_df.merge(snort_df, on=["timestamp", "src_ip", "dest_ip"], how="outer").fillna(0)

# Debug: Print column names to check
print("Columns in merged_df:", merged_df.columns)

# Convert timestamps safely
merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"], errors="coerce")

# Compute time difference safely
merged_df["time_diff"] = (merged_df["timestamp"] - merged_df["timestamp"]).dt.total_seconds()

# Calculate accuracy for anomaly-based IDS
anomaly_TP = (pred_df["prediction"] == 1).sum()
anomaly_TN = (pred_df["prediction"] == 0).sum()
anomaly_accuracy = (anomaly_TP + anomaly_TN) / len(pred_df) if len(pred_df) > 0 else 0

# Calculate accuracy for Snort-based IDS
snort_TP = (snort_df["snort_detected"] == 1).sum()
snort_TN = (snort_df["snort_detected"] == 0).sum()
snort_accuracy = (snort_TP + snort_TN) / len(snort_df) if len(snort_df) > 0 else 0

# Calculate average detection time for both systems
anomaly_avg_time = pred_df["timestamp"].diff().mean().total_seconds()
snort_avg_time = snort_df["timestamp"].diff().mean().total_seconds()

# Print results
print(f"Anomaly-based IDS Accuracy: {anomaly_accuracy:.4f}")
print(f"Snort-based IDS Accuracy: {snort_accuracy:.4f}")
print(f"Anomaly-based IDS Average Detection Time: {anomaly_avg_time:.4f} seconds")
print(f"Snort-based IDS Average Detection Time: {snort_avg_time:.4f} seconds")