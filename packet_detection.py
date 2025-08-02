import joblib
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP

def preprocess_packet(packet_df):
    # Ensure proto is float
    packet_df["proto"] = packet_df["proto"].astype(float)
    
    # One-hot encode proto
    proto_encoded = pd.get_dummies(packet_df["proto"], prefix="proto")
    
    # One-hot encode flags
    if "flags" in packet_df.columns:
        flags_encoded = pd.get_dummies(packet_df["flags"], prefix="flags")
    else:
        flags_encoded = pd.DataFrame()
    
    # Drop original categorical columns
    packet_df = packet_df.drop(columns=["proto", "flags"], errors='ignore')
    
    # Merge encoded features
    packet_df = pd.concat([packet_df, proto_encoded, flags_encoded], axis=1)
    
    return packet_df

def process_packet(packet):
    if IP in packet:
        packet_data = {
            "src_ip": packet[IP].src,
            "dst_ip": packet[IP].dst,
            "proto": float(packet[IP].proto),
            "len": packet[IP].len,
            "sport": packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else 0),
            "dport": packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else 0),
            "flags": packet.sprintf("%TCP.flags%") if TCP in packet else ""
        }
        df = pd.DataFrame([packet_data])
        
        # Preprocess the packet
        processed_df = preprocess_packet(df)
        
        # Align columns with trained model
        missing_cols = [col for col in model.feature_names_in_ if col not in processed_df.columns]
        for col in missing_cols:
            processed_df[col] = 0  # Add missing columns with default 0
        
        # Ensure column order
        processed_df = processed_df[model.feature_names_in_]
        
        # Predict
        prediction = model.predict(processed_df)[0]
        print(f"Packet from {packet_data['src_ip']} to {packet_data['dst_ip']} classified as: {'Threat' if prediction else 'Normal'}")

# Load the trained model
model = joblib.load("random_forest_model.pkl")

print("Starting real-time packet capture...")
sniff(prn=process_packet, store=False)
