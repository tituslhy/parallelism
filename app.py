from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, File, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
import logging

from src.processors import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    
    logger.info("🚀 Starting ingestion API with Qdrant + Ollama...")
    processor = DocumentProcessor()
    await processor.start_workers()
    logger.info("✅ Ingestion API ready!")
    
    yield
    
    # On shutdown
    logger.info("🛑 Shutting down ingestion API...")

app = FastAPI(
    title="Vector DB ingestion API",
    description="Document ingestion with Qdrant + Ollama",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/upload-document/")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = "default_user"
):
    """Upload document for processing"""
    try:
        content = await file.read()
        text_content = content.decode('utf-8')
        
        doc_id = await processor.add_document_to_queue(
            user_id=user_id,
            filename=file.filename,
            content=text_content
        )
        
        return JSONResponse({
            "message": "Document uploaded successfully",
            "document_id": doc_id,
            "status": "queued",
            "filename": file.filename
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/search")
async def semantic_search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=20, description="Number of results")
):
    """Semantic search using Qdrant + Ollama"""
    try:
        results = await processor.semantic_search(query, limit)
        return JSONResponse({
            "query": query,
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/document-status/{doc_id}")
async def get_document_status(doc_id: str):
    """Check document processing status"""
    status_info = await processor.get_document_status(doc_id)
    return JSONResponse(status_info)

@app.get("/health")
async def health_check():
    """Health check - includes Qdrant and Ollama connectivity"""
    try:
        # Check Qdrant
        collections = await processor.qdrant_client.get_collections()
        qdrant_status = "healthy"
        
        # Check Ollama
        test_embedding = await processor.ollama_client.generate_embedding("test")
        ollama_status = "healthy" if test_embedding else "unhealthy"
        
        return JSONResponse({
            "status": "healthy",
            "qdrant": qdrant_status,
            "ollama": ollama_status,
            "workers_active": processor.workers_started
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e)
        }, status_code=503)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)