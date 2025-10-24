# main.py
import os
import shutil
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException,APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
# --- Local imports for utils ---
from .utils.extract_text import extract_text
from .utils.embed_store import init_collection, store_embeddings, query_embeddings


# --- Router imports ---
from backend.routes import documents, chat, users, apikeys

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- CONFIG ---
UPLOAD_DIR = "../../backend/public/uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/responses"

# --- FastAPI instance ---
app = FastAPI(title="RAG Document Service")

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use origins=origins for whitelist
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include routers ---
app.include_router(users.router, prefix="/auth")
app.include_router(chat.router)
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(apikeys.router, prefix="/apikeys", tags=["API Keys"])

# --- Global variable to track collection initialization ---
COLLECTION_INIT = False

# --- Helper: Initialize Qdrant Collection ---
def ensure_collection():
    global COLLECTION_INIT
    if not COLLECTION_INIT:
        init_collection()
        COLLECTION_INIT = True

# --- Helper: Validate required fields ---
def validate_fields(fields: dict):
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

# --- Endpoints: Upload & Embed / Query / Embed Existing ---
@app.post("/upload_and_embed")
async def upload_and_embed(
    user_id: str = Form(...),
    api_key: str = Form(...),
    # doc_id: str = Form(...),
    file: UploadFile = File(...)
):
    validate_fields({"user_id": user_id, "api_key": api_key, "file": file.filename})
    ensure_collection()

    file_path = os.path.join(UPLOAD_DIR, f"{user_id}_{file.filename}")
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        text = extract_text(file_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {e}")

    chunk_size = 2000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    try:
        store_embeddings(user_id, api_key, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding/storage failed: {e}")

    return {
        "message": "Document processed successfully",
        "user_id": user_id,
        "num_chunks": len(chunks),
        "file_path": file_path
    }

@app.post("/query_llm")
async def query_llm(
    user_id: str = Form(...),
    query: str = Form(...)
):
    print(GROQ_API_KEY)
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Set GROQ_API_KEY in environment.")
    print(user_id+" "+query,end="\n")
    ensure_collection()
    # doc_list = [d.strip() for d in doc_ids.split(",") if d.strip()]
    chunks = query_embeddings(user_id=user_id, query_text=query)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context found in vector DB.")

    context_text = "\n\n".join(chunks)
    prompt = f"""
        You are a helpful assistant. Use the context below to answer the question.No other than the context is allowed to be used to answer the question.
        Provide only **one to four concise sentence** as the answer. Do NOT include reasoning, steps, or extra text.

        Context:
        {context_text}

        Question: {query}
        Answer:
    """

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "input": prompt,
        "temperature": 0.2,
        "max_output_tokens": 512
    }

    resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
    # if resp.status_code != 200:
    #     raise HTTPException(status_code=500, detail=f"Groq API call failed: {resp.status_code} {resp.text}")
    if resp.status_code != 200:
        error_text = resp.text

        if "Request too large" in error_text or "413" in error_text:
            return {
                "llm_response": "⚠ Response too large for this plan (limit 6000 tokens). Please shorten your input or upgrade your Groq plan."
            }
        elif "rate_limit_exceeded" in error_text:
            return {
                "llm_response": "⚠ Rate limit exceeded. Please wait a few seconds and try again."
            }
        elif "limit 6000" in error_text:
            return {
                "llm_response": "⚠ Request exceeded 6000 token limit. Try reducing context or question size."
            }
        else:
            raise HTTPException(status_code=500, detail=f"Groq API call failed: {resp.status_code} {error_text}")


    result = resp.json()
    output_texts = []
    for item in result.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                output_texts.append(c.get("text").strip())

    llm_response = " ".join(output_texts).strip()

    return {"llm_response": llm_response}


router = APIRouter()

# --- Dependency: Validate API Key ---
def validate_api_key(api_key: str = Form(...)):
    """
    This function should call your API key authentication service.
    For now, we assume all keys starting with 'kn_' are valid.
    Replace with real auth provider integration.
    """
    if not api_key.startswith("kn_"):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

@app.post("/query_llm_api")
async def query_llm_api(
    query: str = Form(...),
    api_key: str = Depends(validate_api_key)
):
    """
    External API endpoint for LLM query using API Key authentication.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: GROQ_API_KEY not set")

    # 1️⃣ Fetch relevant chunks from vector DB based on api_key
    chunks = query_embeddings(api_key=api_key, query_text=query)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context found in vector DB.")

    context_text = "\n\n".join(chunks)

    # 2️⃣ Prepare prompt for LLM
    prompt = f"""
        You are a helpful assistant. Use the context below to answer the question.No other than the context is allowed to be used to answer the question.
        Provide only one to four concise sentences as the answer. Do NOT include reasoning, steps, or extra text.

        Context:
        {context_text}

        Question: {query}
        Answer:
    """

    # 3️⃣ Call Groq LLM
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "input": prompt,
        "temperature": 0.2,
        "max_output_tokens": 512
    }

    resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
    if resp.status_code != 200:
        error_text = resp.text

        if "Request too large" in error_text or "413" in error_text:
            return {
                "llm_response": "⚠ Response too large for this plan (limit 6000 tokens). Please shorten your input or upgrade your Groq plan."
            }
        elif "rate_limit_exceeded" in error_text:
            return {
                "llm_response": "⚠ Rate limit exceeded. Please wait a few seconds and try again."
            }
        elif "limit 6000" in error_text:
            return {
                "llm_response": "⚠ Request exceeded 6000 token limit. Try reducing context or question size."
            }
        else:
            raise HTTPException(status_code=500, detail=f"Groq API call failed: {resp.status_code} {error_text}")


    result = resp.json()
    output_texts = [
        c.get("text").strip()
        for item in result.get("output", [])
        for c in item.get("content", [])
        if c.get("type") == "output_text"
    ]

    llm_response = " ".join(output_texts).strip()
    return {"llm_response": llm_response}

@app.post("/embed_existing")
async def embed_existing(
    user_id: str = Form(...),
    api_key: str = Form(...),
    filename: str = Form(...)
):
    validate_fields({"user_id": user_id, "api_key": api_key, "filename": filename})
    ensure_collection()

    file_path = os.path.join(UPLOAD_DIR, f"{user_id}_{filename}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        text = extract_text(file_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {e}")

    chunk_size = 2000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    try:
        store_embeddings(user_id, api_key, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding/storage failed: {e}")

    return {
        "message": "Document embedded successfully",
        "user_id": user_id,
        "num_chunks": len(chunks),
        "file_path": file_path
    }

# --- Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "collection_initialized": COLLECTION_INIT}

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from backend.routes import documents, chat, users , apikeys

# app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
#     "http://127.0.0.1",
#     "http://127.0.0.1:3000",
# ]

# # Add CORS Middleware here before your routes
# app.add_middleware(
#     CORSMiddleware,
#     # allow_origins=origins,           # whitelist your frontend origins
#     allow_origins=["*"],           # whitelist your frontend origins
#     allow_credentials=True,
#     allow_methods=["*"],             # allow all HTTP methods
#     allow_headers=["*"],             # allow all headers
# )

# # Include routers
# app.include_router(users.router, prefix="/auth")  # Include user auth routes prefixed by /auth
# # app.include_router(documents.router)
# app.include_router(chat.router)
# app.include_router(documents.router, prefix="/documents", tags=["Documents"])
# app.include_router(apikeys.router, prefix="/apikeys", tags=["API Keys"])