from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pipeline
import os
import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_activity.log")
    ]
)
logger = logging.getLogger("RAG-API")

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
    os.makedirs("data", exist_ok=True)
    logger.info("Server started. Waiting for PDF upload via /upload endpoint.")
    yield
    logger.info("Shutting down API server.")

app = FastAPI(
    title="Hybrid RAG API",
    description="Advanced RAG API powered by FAISS, BM25, Cross-Encoders, and Groq.",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Hybrid RAG API"}

@app.get("/document")
async def get_current_document():
    pdf_path = os.getenv("RAG_PDF_PATH", "")
    if not pdf_path:
        return {"current_document": None, "message": "No document uploaded yet."}
    return {"current_document": os.path.basename(pdf_path)}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join("data", file.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file.")
    finally:
        file.file.close()

    logger.info(f"New PDF uploaded: {file.filename}. Indexing...")

    try:
        from dense_retrieval import clear_index
        clear_index()
        pipeline.initialize(pdf_path=save_path)
        os.environ["RAG_PDF_PATH"] = save_path
        logger.info(f"Successfully indexed: {file.filename}")
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
    if not os.getenv("RAG_PDF_PATH"):
        raise HTTPException(status_code=400, detail="No document uploaded yet. Please upload a PDF first via /upload.")

    logger.info(f"Received query: '{request.question}'")
    try:
        result = pipeline.ask(
            question=request.question,
            hybrid_top_k=request.hybrid_top_k,
            rerank_top_k=request.rerank_top_k
        )
        logger.info(f"Answered in {result['total_time']:.2f}s")
        return QueryResponse(**result)
    except RuntimeError as e:
        logger.error(f"RuntimeError during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)