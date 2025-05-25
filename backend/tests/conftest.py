import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app

@pytest.fixture
def client():
    """
    Test client fixture for FastAPI application
    """
    with TestClient(app) as test_client:
        yield test_client
