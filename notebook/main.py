from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json
import os
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Support Ticket Classifier API",
    description="AI-powered support ticket classification with confidence scores",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
label2id = None
id2label = None

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., description="Support ticket text to classify", min_length=1, max_length=5000)
    include_all_scores: Optional[bool] = Field(default=False, description="Include confidence scores for all classes")

class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted support ticket category")
    confidence: float = Field(..., description="Confidence score for the prediction (0-1)")
    all_scores: Optional[Dict[str, float]] = Field(None, description="Confidence scores for all classes")
    processing_time_ms: float = Field(..., description="Time taken to process the request in milliseconds")
    timestamp: str = Field(..., description="Timestamp of the prediction")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of support ticket texts to classify", max_items=100)
    include_all_scores: Optional[bool] = Field(default=False, description="Include confidence scores for all classes")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    batch_processing_time_ms: float
    timestamp: str

def load_model(model_path: str = "./support_ticket_classifier_optimized"):
    """Load the trained model and tokenizer"""
    global model, tokenizer, device, label2id, id2label
    
    try:
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer loaded successfully")
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")
        
        # Extract label mappings
        label2id = model.config.label2id
        id2label = model.config.id2label
        
        # Convert id2label keys to integers if they're strings
        if id2label and isinstance(list(id2label.keys())[0], str):
            id2label = {int(k): v for k, v in id2label.items()}
        
        logger.info(f"Label mappings loaded: {id2label}")
        logger.info(f"Model loaded with {len(id2label)} classes")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def clean_text(text: str) -> str:
    """Clean and preprocess text input"""
    text = str(text).lower().strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def predict_single(text: str, include_all_scores: bool = False) -> Dict:
    """Make prediction for a single text input"""
    global model, tokenizer, device, id2label
    
    start_time = datetime.now()
    
    try:
        # Clean the input text
        cleaned_text = clean_text(text)
        
        # Tokenize the input
        inputs = tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)
            confidence_scores = probabilities.cpu().numpy()[0]
            
            # Get predicted class
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            predicted_class = id2label[predicted_class_id]
            confidence = float(confidence_scores[predicted_class_id])
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Prepare response
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        # Include all scores if requested
        if include_all_scores:
            all_scores = {}
            for class_id, score in enumerate(confidence_scores):
                class_name = id2label[class_id]
                all_scores[class_name] = round(float(score), 4)
            result["all_scores"] = all_scores
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Startup event to load model
@app.on_event("startup")
async def startup_event():
    """Load model when the API starts"""
    logger.info("Loading model...")
    success = load_model()
    if not success:
        logger.error("Failed to load model during startup!")
    else:
        logger.info("API ready to serve predictions!")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        timestamp=datetime.now().isoformat()
    )

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """
    Classify a support ticket text and return the predicted category with confidence score.
    
    - **text**: The support ticket text to classify
    - **include_all_scores**: Whether to include confidence scores for all classes
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check the health endpoint.")
    
    try:
        result = predict_single(input_data.text, input_data.include_all_scores)
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    """
    Classify multiple support ticket texts in a single request.
    
    - **texts**: List of support ticket texts to classify (max 100)
    - **include_all_scores**: Whether to include confidence scores for all classes
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check the health endpoint.")
    
    if len(input_data.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts allowed per batch request")
    
    start_time = datetime.now()
    predictions = []
    
    try:
        for text in input_data.texts:
            if not text.strip():
                continue
            result = predict_single(text, input_data.include_all_scores)
            predictions.append(PredictionResponse(**result))
        
        # Calculate total processing time
        total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            batch_processing_time_ms=round(total_processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")

# Get available classes
@app.get("/classes")
async def get_classes():
    """Get all available classification classes"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": list(id2label.values()) if id2label else [],
        "total_classes": len(id2label) if id2label else 0,
        "label_mapping": id2label
    }

# Example endpoint with sample data
@app.get("/example")
async def get_example():
    """Get example requests for testing the API"""
    return {
        "single_prediction": {
            "url": "/predict",
            "method": "POST",
            "example_payload": {
                "text": "My computer is not working properly, can you help me fix it?",
                "include_all_scores": True
            }
        },
        "batch_prediction": {
            "url": "/predict/batch", 
            "method": "POST",
            "example_payload": {
                "texts": [
                    "I can't log into my account",
                    "The website is loading very slowly",
                    "I need help with billing"
                ],
                "include_all_scores": False
            }
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Support Ticket Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict", 
            "batch_predict": "/predict/batch",
            "classes": "/classes",
            "docs": "/docs",
            "example": "/example"
        },
        "status": "ready" if model is not None else "loading"
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",  # Replace "main" with your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )