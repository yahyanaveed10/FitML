from fastapi import APIRouter, Depends
from src.utils.logging.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "healthy", "version": "0.1.0"}
