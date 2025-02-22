from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from twilio.rest import Client
import os
import logging

# Initialize app
app = FastAPI(title="Flood Alert System", version="1.0")

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

# Load ML model (simple version)
try:
    model = torch.load("flood_model.pth")
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Initialize Twilio client if credentials exist
if TWILIO_SID and TWILIO_TOKEN:
    twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
    SMS_ENABLED = True
else:
    SMS_ENABLED = False
    logger.warning("SMS alerts disabled - missing Twilio credentials")

# Request/Response models
class PredictionRequest(BaseModel):
    features: list[float]
    phone_number: str = None

class PredictionResponse(BaseModel):
    probability: float
    uncertainty: float
    alert_sent: bool

# Enable CORS (simplified)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "model_loaded": model is not None,
        "sms_enabled": SMS_ENABLED
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Basic validation
        if len(request.features) != 20:
            raise HTTPException(
                status_code=400, 
                detail="Exactly 20 features required"
            )

        # Convert to tensor
        input_tensor = torch.tensor([request.features], dtype=torch.float32)
        
        # Make predictions
        model.train()  # Enable dropout for uncertainty
        with torch.no_grad():
            predictions = [model(input_tensor).item() for _ in range(20)]
        
        mean_prob = np.mean(predictions)
        std_dev = np.std(predictions)
        
        # Send SMS alert if needed
        alert_sent = False
        if SMS_ENABLED and request.phone_number and mean_prob > 0.7:
            try:
                twilio_client.messages.create(
                    body=f"Flood Alert! Risk: {mean_prob*100:.1f}%",
                    from_=TWILIO_NUMBER,
                    to=request.phone_number
                )
                alert_sent = True
                logger.info(f"Alert sent to {request.phone_number}")
            except Exception as e:
                logger.error(f"Failed to send SMS: {str(e)}")

        return {
            "probability": mean_prob,
            "uncertainty": std_dev,
            "alert_sent": alert_sent
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))