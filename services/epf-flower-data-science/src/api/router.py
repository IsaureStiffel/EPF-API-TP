"""API Router for Fast API."""
from fastapi import APIRouter

from src.api.routes import hello, swagger_doc, data

router = APIRouter()

router.include_router(hello.router, tags=["Hello"])
router.include_router(swagger_doc.router, tags=["Swagger documentation"])
router.include_router(data.router, tags=["Download data"])
