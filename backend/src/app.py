from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from api.routes.api import api_router
from utils.logging.logger import get_logger, setup_file_logging

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger("app")
setup_file_logging(logger)

# Create FastAPI app
app = FastAPI(
    title="ML/RL Backend API",
    description="Backend service for ML and RL operations",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "ML/RL Backend API is running"}

# Include API router
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting application on port {os.getenv('PORT', 8000)}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
