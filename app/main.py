"""
FastAPI application entry point.
Provides REST API endpoints for document upload, search, and RAG.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings

app = FastAPI(
    title="RAG Search Engine",
    description="Production-grade RAG system with advanced NLP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "version": Settings.VERSION}
