"""
Model Registry Service

Handles loading, registering, and managing ML models.
"""
import os
import json
from typing import Dict, List, Any, Optional
import pickle
import tensorflow as tf
import torch
from ...utils.logging.logger import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    """
    Model Registry for managing ML models
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize the model registry
        
        Args:
            model_dir: Directory to store models
        """
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load model metadata if available
        self._load_metadata()
        
        logger.info(f"Model registry initialized with directory: {model_dir}")
    
    def _load_metadata(self) -> None:
        """Load model metadata from the registry"""
        metadata_path = os.path.join(self.model_dir, "registry.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.model_metadata)} models")
            except Exception as e:
                logger.error(f"Failed to load model metadata: {str(e)}")
                self.model_metadata = {}
    
    def _save_metadata(self) -> None:
        """Save model metadata to the registry"""
        metadata_path = os.path.join(self.model_dir, "registry.json")
        
        try:
            with open(metadata_path, "w") as f:
                json.dump(self.model_metadata, f, indent=2)
            logger.info(f"Saved metadata for {len(self.model_metadata)} models")
        except Exception as e:
            logger.error(f"Failed to save model metadata: {str(e)}")
    
    def register_model(self, model_name: str, model_path: str, model_type: str, 
                      version: str, framework: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a model in the registry
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            model_type: Type of model (supervised, reinforcement)
            version: Model version
            framework: Model framework (tensorflow, pytorch, sklearn, etc.)
            metadata: Additional model metadata
            
        Returns:
            True if registration was successful
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Create model entry
        model_info = {
            "path": model_path,
            "type": model_type,
            "version": version,
            "framework": framework,
            "metadata": metadata or {}
        }
        
        # Add to registry
        self.model_metadata[model_name] = model_info
        
        # Save updated metadata
        self._save_metadata()
        
        logger.info(f"Registered model: {model_name} (version: {version})")
        return True
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a model from the registry
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model or None if loading failed
        """
        # Check if model is already loaded
        if model_name in self.models:
            logger.info(f"Using cached model: {model_name}")
            return self.models[model_name]
        
        # Check if model exists in registry
        if model_name not in self.model_metadata:
            logger.error(f"Model not found in registry: {model_name}")
            return None
        
        model_info = self.model_metadata[model_name]
        model_path = model_info["path"]
        framework = model_info["framework"]
        
        try:
            # Load based on framework
            if framework == "tensorflow":
                model = tf.keras.models.load_model(model_path)
            elif framework == "pytorch":
                model = torch.load(model_path)
            elif framework == "sklearn" or framework == "pickle":
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                logger.error(f"Unsupported model framework: {framework}")
                return None
            
            # Cache the model
            self.models[model_name] = model
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return None
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if unloading was successful
        """
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Unloaded model: {model_name}")
            return True
        
        logger.warning(f"Model not loaded, nothing to unload: {model_name}")
        return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in the registry
        
        Returns:
            List of model information dictionaries
        """
        return [
            {
                "name": name,
                "type": info["type"],
                "version": info["version"],
                "framework": info["framework"],
                **info["metadata"]
            }
            for name, info in self.model_metadata.items()
        ]
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information or None if not found
        """
        if model_name not in self.model_metadata:
            return None
        
        info = self.model_metadata[model_name]
        return {
            "name": model_name,
            "type": info["type"],
            "version": info["version"],
            "framework": info["framework"],
            **info["metadata"]
        }

# Singleton instance
_instance = None

def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance
    
    Returns:
        ModelRegistry instance
    """
    global _instance
    
    if _instance is None:
        from src.config.base import Config
        _instance = ModelRegistry(Config.MODEL_DIR)
    
    return _instance
