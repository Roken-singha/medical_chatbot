# Medical Chatbot

A Flask-based medical chatbot leveraging BioBERT and OpenVINO for intelligent question answering in the healthcare domain.

---

## Overview

This project is a medical chatbot designed to provide accurate answers to healthcare-related questions. It uses a fine-tuned BioBERT model, optimized with OpenVINO for efficient inference, and is deployed as a web application using Flask. The chatbot sources questions and contexts from a JSON file (`processed_data.csv`), replacing an earlier CSV-based approach.

### Key Features
- **Question Answering**: Extracts answers from predefined contexts using extractive QA.
- **Model**: Fine-tuned BioBERT (`dmis-lab/biobert-base-cased-v1.1`).
- **Optimization**: OpenVINO for fast CPU inference.
- **Deployment**: Flask app with a RESTful API endpoint (`/generate`).

---
## Setup
1. Install Python 3.8+ and Git.
2. Clone repo: `git clone <url>`
3. Create venv: `python -m venv venv`
4. Activate: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
5. Install deps: `pip install -r requirements.txt`
6. Install OpenVINO Toolkit.

## Run
1. `python src/preprocess.py`
2. `python src/train.py`
3. `python src/optimize.py`
4. `python src/app.py`
5. Open `http://localhost:5000`