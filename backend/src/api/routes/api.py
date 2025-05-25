from fastapi import APIRouter
from ...api.routes import health, ml

api_router = APIRouter()

# Include routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(ml.router, prefix="/ml", tags=["ml"])
