# Hybrid Network Intrusion Detection System

## Overview
This project implements a Hybrid Machine Learning-Based Network Intrusion Detection System combining:

- XGBoost
- Deep Neural Network
- Weighted Probability Fusion

## Features
- Single flow detection
- Batch CSV detection
- Live simulation mode
- REST API backend
- Streamlit dashboard
- Confidence scoring
- Severity classification
- Logging system

## Architecture
- src/ → Core ML engine
- api/ → REST API backend
- dashboard/ → User interface
- models/ → Trained models
- notebooks/ → Research & training

## Run API
python -m uvicorn api.app:app --reload

## Run Dashboard
streamlit run dashboard/app.py

## Model Performance
Hybrid Macro F1: 0.966+
Accuracy: 99.9%D