"""
Model Inference Service

Handles predictions from ML models.
"""
from typing import Dict, List, Any, Union, Optional
import numpy as np
from ..model_registry import get_model_registry
from ....utils.logging.logger import get_logger

logger = get_logger(__name__)

class InferenceService:
    """
    Service for model inference
    """
    
    def __init__(self):
        """Initialize the inference service"""
        self.model_registry = get_model_registry()
        logger.info("Inference service initialized")
    
    def predict(self, model_name: str, features: Union[List[float], np.ndarray], 
               options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a prediction using the specified model
        
        Args:
            model_name: Name of the model to use
            features: Input features for prediction
            options: Additional prediction options
            
        Returns:
            Prediction result
        """
        options = options or {}
        
        # Load the model
        model = self.model_registry.load_model(model_name)
        if model is None:
            logger.error(f"Failed to load model for prediction: {model_name}")
            return {
                "error": f"Model not found: {model_name}",
                "success": False
            }
        
        # Get model info
        model_info = self.model_registry.get_model_info(model_name)
        if model_info is None:
            logger.error(f"Model info not found: {model_name}")
            return {
                "error": f"Model info not found: {model_name}",
                "success": False
            }
        
        # Prepare features
        features_array = np.array(features)
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
        
        try:
            # Make prediction based on framework
            framework = model_info["framework"]
            
            if framework == "tensorflow":
                raw_prediction = model.predict(features_array)
            elif framework == "pytorch":
                import torch
                with torch.no_grad():
                    features_tensor = torch.tensor(features_array, dtype=torch.float32)
                    raw_prediction = model(features_tensor).numpy()
            else:  # sklearn or other frameworks
                raw_prediction = model.predict(features_array)
            
            # Process prediction based on model type
            prediction_result = self._process_prediction(raw_prediction, model_info["type"])
            
            # Add metadata
            prediction_result.update({
                "model_name": model_name,
                "model_version": model_info["version"],
                "success": True
            })
            
            logger.info(f"Prediction successful with model: {model_name}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "error": f"Prediction error: {str(e)}",
                "success": False
            }
    
    def _process_prediction(self, raw_prediction: np.ndarray, model_type: str) -> Dict[str, Any]:
        """
        Process raw prediction into appropriate format
        
        Args:
            raw_prediction: Raw prediction from model
            model_type: Type of model
            
        Returns:
            Processed prediction
        """
        if model_type == "supervised":
            # Classification
            if len(raw_prediction.shape) > 1 and raw_prediction.shape[1] > 1:
                # Multi-class classification
                prediction_class = np.argmax(raw_prediction, axis=1)[0]
                probability = float(raw_prediction[0][prediction_class])
                return {
                    "prediction": int(prediction_class),
                    "probability": probability,
                    "probabilities": raw_prediction[0].tolist()
                }
            else:
                # Binary classification or regression
                prediction = float(raw_prediction[0][0] if len(raw_prediction.shape) > 1 else raw_prediction[0])
                return {
                    "prediction": prediction,
                    "probability": 1.0  # For regression, this doesn't apply
                }
        elif model_type == "reinforcement":
            # RL prediction (e.g., policy output)
            if len(raw_prediction.shape) > 1:
                # Action probabilities
                return {
                    "action": int(np.argmax(raw_prediction[0])),
                    "action_probabilities": raw_prediction[0].tolist()
                }
            else:
                # Single action
                return {
                    "action": int(raw_prediction[0]),
                }
        else:
            # Default fallback
            return {
                "prediction": raw_prediction.tolist()
            }

# Singleton instance
_instance = None

def get_inference_service() -> InferenceService:
    """
    Get the global inference service instance
    
    Returns:
        InferenceService instance
    """
    global _instance
    
    if _instance is None:
        _instance = InferenceService()
    
    return _instance
