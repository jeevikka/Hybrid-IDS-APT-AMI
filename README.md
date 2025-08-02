
# Hybrid IDS for Detecting APTs in AMI

This project implements a hybrid Intrusion Detection System (IDS) for detecting Advanced Persistent Threats (APTs) in Advanced Metering Infrastructure (AMI) networks. It combines machine learning models with traditional signature-based detection (Snort) for higher accuracy and robustness.

## Features
Random Forest + XGBoost-based anomaly detection
Real-time packet inspection using Scapy
Trained on 5 lakh-packet custom dataset
SHAP explainability for model transparency
Snort integration for signature-based detection
Deployed on Raspberry Pi

## Technologies Used
Python (Scikit-learn, XGBoost, SHAP)
Snort
Scapy
Raspberry Pi OS

## Files
ids_model.py`: ML model training and evaluation
packet_capture.py`: Real-time packet capture
snort_simple.conf`: Snort rule config
explainer.ipynb`: SHAP explainability (optional)

## Results
Achieved 92% accuracy and 0.97 AUC
Reduced false positives with hybrid approach
