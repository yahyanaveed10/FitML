from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from ...utils.logging.logger import get_logger
from ...services.ml.model_registry import get_model_registry
from ...services.ml.inference.inference_service import get_inference_service

router = APIRouter()
logger = get_logger(__name__)

class PredictionRequest(BaseModel):
    """Prediction request model"""
    features: List[float]
    model_name: str = "default"
    options: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    """Prediction response model"""
    prediction: float
    probability: float
    model_name: str
    model_version: str
    success: bool

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction using an ML model"""
    logger.info(f"Prediction requested with model {request.model_name}")
    
    # Get inference service
    inference_service = get_inference_service()
    
    # Make prediction
    result = inference_service.predict(
        model_name=request.model_name,
        features=request.features,
        options=request.options
    )
    
    if not result.get("success", False):
        logger.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
        raise HTTPException(
            status_code=500, 
            detail=result.get("error", "Prediction failed")
        )
    
    logger.info(f"Prediction successful: {result.get('prediction')}")
    return result

@router.get("/models")
async def list_models():
    """List available ML models"""
    logger.info("Model list requested")
    
    # Get model registry
    model_registry = get_model_registry()
    
    # Get models
    models = model_registry.list_models()
    
    return {"models": models}

@router.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    logger.info(f"Model info requested: {model_name}")
    
    # Get model registry
    model_registry = get_model_registry()
    
    # Get model info
    model_info = model_registry.get_model_info(model_name)
    
    if model_info is None:
        logger.warning(f"Model not found: {model_name}")
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    
    return model_info
