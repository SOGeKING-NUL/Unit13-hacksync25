from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch.nn as nn
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uvicorn
import torch

load_dotenv()

# Architecture
class FloodPred(nn.Module):
    def __init__(self, input_size=21):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Keep dropout for MC sampling
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
        
    def forward(self, x):
        return self.layers(x)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML Model
try:
    model = FloodPred()
    model.load_state_dict(torch.load("models/flood_model.pth"))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

class PredictionRequest(BaseModel):
    features: list[float]
    phone_number: str = None

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        if len(request.features) != 20:
            raise HTTPException(status_code=400, detail="Exactly 20 features required")
        
        input_tensor = torch.tensor([request.features], dtype=torch.float32)
        
        model.train()  # Enable dropout
        with torch.no_grad():
            predictions = [model(input_tensor).item() for _ in range(20)]
        
        return {
            "probability": float(np.mean(predictions)),
            "uncertainty": float(np.std(predictions)),
            "alert": "High Risk" if np.mean(predictions) > 0.7 else "Moderate Risk"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)