# backend/app.py
from flask import Flask, request, jsonify
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from twilio.rest import Client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ----- PyTorch Model Definition -----
class FloodPred(nn.Module):
    def __init__(self, input_size=20):
        super(FloodPred, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


# ----- Model Loading with Security -----
MODEL_PATHS = {
    'xgboost': 'models/floodXgBoostV1.pkl',
    'pytorch': 'models/floodNN.pt'
}

def load_models():
    """Safely load ML models with error handling"""
    try:
        # Load XGBoost model
        xgboost_model = joblib.load(MODEL_PATHS['xgboost'])
        
        # Load PyTorch model
        pytorch_model = FloodPred(input_size=20)
        pytorch_model.load_state_dict(
            torch.load(MODEL_PATHS['pytorch'], map_location=torch.device('cpu'))
        )
        pytorch_model.eval()
        
        return xgboost_model, pytorch_model

    except FileNotFoundError as e:
        raise RuntimeError(f"Model file not found: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

# Initialize models
try:
    xgboost_model, pytorch_model = load_models()
except RuntimeError as e:
    app.logger.error(str(e))
    exit(1)

#Alert System Configuration 
class DynamicThresholdCalculator:
    def __init__(self, data_path='dataset/train_data.csv'):
        try:
            self.data = pd.read_csv(data_path)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        except Exception as e:
            app.logger.error(f"Error loading threshold data: {str(e)}")
            self.data = pd.DataFrame()

    def calculate_threshold(self):
    
        if self.data.empty:
            return 0.45  # Lower fallback threshold
    
        time_window = datetime.now() - timedelta(hours=6)  # Shorter window
        recent_data = self.data[self.data['timestamp'] > time_window]
    
        if len(recent_data) > 50:
            return recent_data['risk_score'].quantile(0.75)
        return 0.5

# Initialize systems
threshold_calculator = DynamicThresholdCalculator()
twilio_client = Client(os.getenv('TWILIO_ACCOUNT_SID'), 
                      os.getenv('TWILIO_AUTH_TOKEN'))

# ----- API Endpoints -----
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features', [])
        phone_number = data.get('phone', '')
        location = data.get('location', 'Unknown')
        
        if len(features) != 20:
            return jsonify({'error': 'Exactly 20 features required'}), 400

        # Static threshold configuration
        STATIC_THRESHOLD = 0.65  # 65% risk threshold
        
        # Model predictions
        xgb_pred = xgboost_model.predict(np.array([features]))[0]
        with torch.no_grad():
            torch_pred = pytorch_model(torch.tensor([features], dtype=torch.float32)).item()
            
        combined_risk = (xgb_pred * 0.7 + torch_pred * 0.3)
        
        # Determine alert status
        alert_status = "High Risk" if combined_risk > STATIC_THRESHOLD else "Low Risk"
        
        # SMS Alert Logic
        sms_sent = False
        if alert_status == "High Risk" and phone_number:
            try:
                message = f"""🚨 FLOOD ALERT 🚨
Location: {location}
Risk Level: {combined_risk:.0%}
Immediate Action Required!"""
                
                twilio_client.messages.create(
                    body=message,
                    from_=os.getenv('TWILIO_PHONE'),
                    to=phone_number
                )
                sms_sent = True
            except Exception as e:
                app.logger.error(f"SMS Failed: {str(e)}")

        return jsonify({
            'risk': combined_risk,
            'threshold': STATIC_THRESHOLD,
            'alert': alert_status,
            'sms_sent': sms_sent
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'operational',
        'models': list(MODEL_PATHS.keys()),
        'threshold_data': not threshold_calculator.data.empty
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
