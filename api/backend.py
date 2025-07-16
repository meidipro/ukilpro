# backend.py

import os
import shutil
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your powerful RAG engine
from rag_pipeline import MultiModalRAGPipeline

# --- Configuration and Initialization ---

# Define directories for storing user-specific data
# In a real production environment, you might use cloud storage (S3, GCS)
UPLOAD_DIRECTORY = Path("user_uploads")
DB_DIRECTORY = Path("user_dbs")

# Create directories on startup if they don't exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(DB_DIRECTORY, exist_ok=True)

# Initialize the FastAPI app
app = FastAPI(
    title="Legal AI Backend",
    description="API for uploading, analyzing, and querying legal documents.",
    version="1.0.0"
)

# Add CORS middleware to allow requests from your frontend
# (e.g., a React/Vue/Svelte app running on localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a single, shared instance of your RAG engine.
# The engine is stateless regarding user data, which we will manage per request.
try:
    rag_engine = MultiModalRAGPipeline()
    print("MultiModalRAGPipeline initialized successfully.")
except Exception as e:
    print(f"FATAL: Could not initialize RAG Pipeline: {e}")
    # In a real app, you might exit or prevent the app from starting
    rag_engine = None

# --- Pydantic Models for Request Bodies ---

class QueryRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    filename: str

# --- API Endpoints ---

@app.get("/", tags=["Status"])
async def root():
    """Root endpoint to check if the service is running."""
    return {"message": "Legal AI Backend is running!", "status": "OK" if rag_engine else "ERROR"}


@app.post("/upload-document/{user_id}", tags=["Documents"])
async def upload_document(user_id: str, file: UploadFile = File(...)):
    """
    Upload a document for a specific user.
    The document is saved and processed into a persistent vector store.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is not available.")

    user_upload_dir = UPLOAD_DIRECTORY / user_id
    os.makedirs(user_upload_dir, exist_ok=True)
    
    save_path = user_upload_dir / file.filename
    
    try:
        # Save the uploaded file to the user's directory
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Define the path for the user's persistent vector database
        user_db_path = str(DB_DIRECTORY / user_id)
        
        print(f"Processing file '{save_path}' for user '{user_id}' into db '{user_db_path}'")
        
        # Use the engine to process the document into the user's DB
        # The engine handles loading, splitting, embedding, and storing
        processing_result = rag_engine.process_documents(
            file_paths=[str(save_path)],
            persist_directory=user_db_path
        )
        
        return JSONResponse(content={
            "message": f"Document '{file.filename}' uploaded and processed successfully!",
            "user_id": user_id,
            "filename": file.filename,
            **processing_result
        })
        
    except Exception as e:
        # Log the full error for debugging
        print(f"Error processing file for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/analyze-document/{user_id}", tags=["Analysis"])
async def analyze_document(user_id: str, request: DocumentRequest):
    """
    Perform a specialized legal analysis of a previously uploaded document.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is not available.")
        
    doc_path = UPLOAD_DIRECTORY / user_id / request.filename
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{request.filename}' not found for user '{user_id}'.")
        
    try:
        # 1. Load the document content using the RAG engine's loader
        docs = rag_engine.load_document(str(doc_path))
        full_text = "\n".join([doc.page_content for doc in docs])
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text content could be extracted from the document.")
            
        # 2. Use the specialized prompt for Bangladesh legal context
        docu_reviewer_instruction = """
        You are an expert legal document reviewer specializing in Bangladesh law and legal system. 
        Your task is to provide comprehensive analysis of legal documents with focus on:

        1. Document Classification: Identify the type of legal document (contract, agreement, legal notice, etc.)
        2. Legal Compliance: Check compliance with Bangladesh laws and regulations.
        3. Key Legal Points: Highlight important legal clauses, terms, and conditions.
        4. Risk Assessment: Identify potential legal risks, loopholes, or problematic clauses.
        5. Recommendations: Provide actionable recommendations for improvement or compliance.
        6. Language Analysis: Check for legal terminology accuracy and clarity.
        7. Structural Analysis: Evaluate document structure and completeness.

        Please provide your analysis in a structured format with clear headings and bullet points.
        Be thorough but concise, focusing on practical legal insights.
        """
        
        # 3. Create the analysis chain using the engine's LLM instance
        from langchain_core.prompts import ChatPromptTemplate
        
        document_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", docu_reviewer_instruction),
            ("user", "Document: {document}\n\nPlease provide a comprehensive analysis of the document."),
        ])
        
        analysis_chain = document_analysis_prompt | rag_engine.llm
        
        # 4. Invoke the chain and get the response
        response = analysis_chain.invoke({"document": full_text})
        
        return JSONResponse(content={"analysis": response.content})
        
    except Exception as e:
        print(f"Error analyzing document for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze document: {str(e)}")


@app.post("/query-document/{user_id}", tags=["Query"])
async def query_document(user_id: str, request: QueryRequest):
    """
    Ask a question about the documents uploaded by a specific user.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is not available.")

    user_db_path = str(DB_DIRECTORY / user_id)
    if not os.path.exists(user_db_path):
        raise HTTPException(status_code=404, detail="No documents have been processed for this user. Please upload a document first.")
        
    try:
        # This is a critical step for multi-user support:
        # We re-initialize the vector store and retrieval chain for the specific user's DB.
        # This ensures we are querying the correct data.
        rag_engine.create_vector_store([], persist_directory=user_db_path) # Load existing DB
        rag_engine.setup_retrieval_chain()
        
        # Now, query using the fully configured chain
        response = rag_engine.query(request.query)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error querying document for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query document: {str(e)}")