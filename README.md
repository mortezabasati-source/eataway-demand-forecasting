# Eataway Demand Forecasting System 🚀

This repository contains the core machine learning pipeline for **Eataway**, designed to accurately predict weekly product demand across multiple stores and routes.

## 📌 Overview

The system uses historical order and return data (pulled from a MySQL database) to generate intelligent forecasts. The pipeline addresses complex business challenges such as:
- **Zero-inflated demand:** Addressed using a dual-stage Calibrated Hurdle Model and a Tweedie Regressor.
- **Delivery Day Patterns:** Dynamically allocating weekly demand to specific delivery days based on each store's unique schedule.
- **Seasonality & Events:** Sophisticated handling of Swedish holidays and seasonal variations.

## 🏗 System Architecture

The pipeline consists of three main components:

1. **`feature.py` - Data Processing & Feature Engineering**
   - Connects to the Eataway MySQL database.
   - Cleans data (removes discontinued products, handles truncated weeks).
   - Generates lag features, rolling windows, target encodings (leakage-free), and holiday interaction features.

2. **`eataway_train_v7.py` - Model Training & Prediction**
   - Trains a **V7 Ensemble Model** (Calibrated Hurdle + Tweedie).
   - Applies global scaling and bias correction to prevent underestimation.
   - Generates two primary views:
     - **Kitchen View:** Aggregated demand by product for production planning.
     - **Driver View:** Store-level routing and delivery quantities.
   - Automatically exports predictions directly to **Google Sheets** for operational use.

3. **`auto_sync.py` & `run_sync.bat` - Automation**
   - Designed to run via Windows Task Scheduler.
   - Executes the data extraction, training, and Google Sheet export sequence automatically.

## 🚀 How to Run Locally

### 1. Requirements
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
You must provide your database credentials and Google Service Account key. 
Create a `.env` file in the root directory:
```ini
DB_HOST=your_host
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=your_database
```
Place your Google Service Account key file in the root directory and name it `credentials.json`.

*(Note: Both `.env` and `credentials.json` are excluded via `.gitignore` for security).*

### 3. Execution
Run the full pipeline (data extraction → training → Google Sheets export) automatically:
```bash
python auto_sync.py
```
Or via the batch file:
```cmd
run_sync.bat
```

## 🔒 Security & Deployment Notes
This repository is optimized for **Production/Local execution**. 
- All sensitive credentials, auto-generated `.csv` datasets, and model artifacts (`output_v7/`, `predictions/`) are excluded from version control to keep the repository lightweight and secure.
- The pipeline determines the **Target Week** automatically. If run on a Sunday, it predicts the current week. If run on a Saturday, it predicts the upcoming week.
