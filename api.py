from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pipeline
import os
import logging
import time
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_activity.log")
    ]
)
logger = logging.getLogger("RAG-API")

# Define Request and Response structures
class QueryRequest(BaseModel):
    question: str
    hybrid_top_k: int = 20
    rerank_top_k: int = 3

class SourceChunk(BaseModel):
    page: int
    rerank_score: float | None = None
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieval_time: float
    generation_time: float
    total_time: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager runs before the server starts accepting requests.
    It parses the PDF and builds the Vector/BM25 indices in memory.
    """
    pdf_path = os.getenv("RAG_PDF_PATH", "data/sample.pdf")
    logger.info(f"Initializing RAG pipeline with {pdf_path}...")
    try:
        pipeline.initialize(pdf_path=pdf_path)
        logger.info("Pipeline initialized successfully! Ready for requests.")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise e
    yield
    logger.info("Shutting down API server.")

app = FastAPI(
    title="Hybrid RAG API",
    description="Advanced RAG API powered by FAISS, BM25, Cross-Encoders, and Groq.",
    version="1.0.0",
    lifespan=lifespan
)

# Serve the frontend chat UI
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Monitoring endpoint for system health and uptime."""
    return {"status": "healthy", "service": "Hybrid RAG API"}


@app.get("/document")
async def get_current_document():
    """Returns the name of the currently indexed PDF document."""
    pdf_path = os.getenv("RAG_PDF_PATH", "data/sample.pdf")
    return {"current_document": os.path.basename(pdf_path)}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a new PDF to replace the currently indexed document.
    The old index is cleared and the new PDF is fully re-indexed.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded PDF to the data directory
    save_path = os.path.join("data", file.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file.")
    finally:
        file.file.close()

    logger.info(f"New PDF uploaded: {file.filename}. Clearing old index and re-indexing...")

    # Clear old ChromaDB index and re-initialize with new PDF
    try:
        from dense_retrieval import clear_index
        clear_index()
        pipeline.initialize(pdf_path=save_path)
        os.environ["RAG_PDF_PATH"] = save_path
        logger.info(f"Successfully re-indexed: {file.filename}")
    except Exception as e:
        logger.error(f"Failed to index uploaded PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {str(e)}")

    return {
        "message": f"Successfully uploaded and indexed '{file.filename}'",
        "document": file.filename,
        "path": save_path
    }

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Submit a natural language question to the RAG pipeline.
    """
    logger.info(f"Received query: '{request.question}'")
    try:
        result = pipeline.ask(
            question=request.question,
            hybrid_top_k=request.hybrid_top_k,
            rerank_top_k=request.rerank_top_k
        )
        logger.info(f"Answered in {result['total_time']:.2f}s (Retrieval: {result['retrieval_time']:.2f}s, Generation: {result['generation_time']:.2f}s)")
        return QueryResponse(**result)
    except RuntimeError as e:
        logger.error(f"RuntimeError during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    # Start the development server
    uvicorn.run(app, host="127.0.0.1", port=8000)
