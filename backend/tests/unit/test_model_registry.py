import os
import pytest
import numpy as np
from ...src.services.ml.model_registry import ModelRegistry
from ...src.services.ml.inference.inference_service import InferenceService

@pytest.fixture
def model_registry():
    """Model registry fixture"""
    # Use a test directory
    registry = ModelRegistry("./models")
    return registry

@pytest.fixture
def inference_service(model_registry):
    """Inference service fixture"""
    service = InferenceService()
    # Override the model registry
    service.model_registry = model_registry
    return service

def test_list_models(model_registry):
    """Test listing models"""
    models = model_registry.list_models()
    assert isinstance(models, list)
    
    # If we have any models, check their structure
    if models:
        model = models[0]
        assert "name" in model
        assert "type" in model
        assert "version" in model

def test_get_model_info(model_registry):
    """Test getting model info"""
    # This test assumes the linear_regression model exists
    # Skip if registry is empty
    if not model_registry.model_metadata:
        pytest.skip("No models in registry")
        
    model_name = next(iter(model_registry.model_metadata.keys()))
    info = model_registry.get_model_info(model_name)
    
    assert info is not None
    assert "name" in info
    assert info["name"] == model_name

def test_load_model(model_registry):
    """Test loading a model"""
    # This test assumes the linear_regression model exists
    # Skip if registry is empty
    if not model_registry.model_metadata:
        pytest.skip("No models in registry")
        
    model_name = next(iter(model_registry.model_metadata.keys()))
    model = model_registry.load_model(model_name)
    
    assert model is not None
    
    # Test prediction
    if hasattr(model, "predict"):
        # Simple test input
        X = np.array([[1.0, 2.0]])
        prediction = model.predict(X)
        assert prediction is not None

def test_inference_service(inference_service):
    """Test inference service"""
    # This test assumes the linear_regression model exists
    # Skip if no models
    if not inference_service.model_registry.model_metadata:
        pytest.skip("No models in registry")
        
    model_name = next(iter(inference_service.model_registry.model_metadata.keys()))
    
    # Make a prediction
    result = inference_service.predict(
        model_name=model_name,
        features=[1.0, 2.0]
    )
    
    assert result is not None
    assert "success" in result
    assert result["success"] is True
    assert "prediction" in result
