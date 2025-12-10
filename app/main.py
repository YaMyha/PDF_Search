import uvicorn
import re
import secrets
import time
from datetime import timedelta
from typing import Optional
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Form, Query, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.orm import Session
import redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from app.models import SessionLocal, Document, User
from app.rag import answer_question, save_pdf, get_chroma
from app.auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_user, get_db, ACCESS_TOKEN_EXPIRE_MINUTES
)

# Security constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_QUESTION_LENGTH = 2000
MAX_DOC_ID_LENGTH = 100
MAX_USERNAME_LENGTH = 50
MAX_FILENAME_LENGTH = 255

# Rate limiting constants
RATE_LIMIT_ATTEMPTS = 5  # Max attempts
RATE_LIMIT_WINDOW = 300  # 5 minutes in seconds
RATE_LIMIT_LOCKOUT = 900  # 15 minutes lockout after max attempts

# Redis connection
redis_client: Optional[redis.Redis] = None

def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client instance"""
    return redis_client

def init_redis():
    """Initialize Redis connection"""
    global redis_client
    try:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False,  # We'll handle encoding manually for timestamps
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        # Test connection
        redis_client.ping()
        print(f"✓ Redis connected to {redis_host}:{redis_port}")
    except (RedisConnectionError, RedisError) as e:
        print(f"⚠ Redis connection failed: {e}. Rate limiting will be disabled.")
        redis_client = None
    except Exception as e:
        print(f"⚠ Redis initialization error: {e}. Rate limiting will be disabled.")
        redis_client = None

def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        try:
            redis_client.close()
            print("✓ Redis connection closed")
        except Exception as e:
            print(f"⚠ Error closing Redis connection: {e}")
        finally:
            redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    # Startup
    init_redis()
    yield
    # Shutdown
    close_redis()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="app/templates")

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Pydantic models for request/response
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=MAX_USERNAME_LENGTH, pattern="^[a-zA-Z0-9_]+$")
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password has sufficient complexity"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r'[A-Za-z]', v):
            raise ValueError("Password must contain at least one letter")
        if not re.search(r'[0-9]', v):
            raise ValueError("Password must contain at least one number")
        return v


class UserResponse(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


def get_client_ip(request: Request) -> str:
    """Get client IP address for rate limiting"""
    # Check for forwarded IP (when behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(identifier: str) -> bool:
    """
    Check if request should be rate limited using Redis.
    Returns True if allowed, False if blocked.
    Falls back to allowing requests if Redis is unavailable.
    """
    client = get_redis_client()
    if not client:
        # If Redis is unavailable, allow the request (fail open)
        # In production, you might want to fail closed instead
        return True
    
    try:
        now = time.time()
        key = f"ratelimit:{identifier}"
        lockout_key = f"ratelimit:lockout:{identifier}"
        
        # Check if still in lockout period
        lockout_until_bytes = client.get(lockout_key)
        if lockout_until_bytes:
            lockout_until = float(lockout_until_bytes)
            if lockout_until > now:
                return False
            # Lockout expired, remove it
            client.delete(lockout_key)
        
        # Use Redis sorted set to track attempts with timestamps
        # Remove old attempts outside the window
        window_start = now - RATE_LIMIT_WINDOW
        client.zremrangebyscore(key, 0, window_start)
        
        # Count current attempts in the window
        attempt_count = client.zcard(key)
        
        # Check if limit exceeded
        if attempt_count >= RATE_LIMIT_ATTEMPTS:
            # Set lockout period
            client.setex(lockout_key, int(RATE_LIMIT_LOCKOUT), str(now + RATE_LIMIT_LOCKOUT))
            return False
        
        # Record this attempt (add to sorted set with current timestamp as score)
        # Use member as unique identifier (timestamp as string) and score as timestamp
        # For redis-py >= 4.0, use mapping format
        client.zadd(key, {str(now): now})
        # Set expiration on the key to auto-cleanup (window + lockout time)
        client.expire(key, int(RATE_LIMIT_WINDOW + RATE_LIMIT_LOCKOUT))
        
        return True
    except (RedisError, ValueError, TypeError) as e:
        # Log error but allow request (fail open)
        # In production, consider logging to monitoring system
        print(f"⚠ Redis rate limit error for {identifier}: {e}")
        return True


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister, request: Request, db: Session = Depends(get_db)):
    """Register a new user"""
    # Rate limiting
    client_ip = get_client_ip(request)
    if not check_rate_limit(f"register:{client_ip}"):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Please try again later."
        )
    
    # Check if username already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user (password validation is done by Pydantic)
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user


@app.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Authenticate user and return access token"""
    # Rate limiting by IP and username
    client_ip = get_client_ip(request)
    rate_limit_key = f"login:{client_ip}:{form_data.username}"
    
    if not check_rate_limit(rate_limit_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and injection attacks"""
    if not filename:
        return "document.pdf"
    # Remove path components
    filename = os.path.basename(filename)
    # Remove any remaining path separators
    filename = filename.replace("/", "").replace("\\", "")
    # Limit length
    if len(filename) > MAX_FILENAME_LENGTH:
        name, ext = os.path.splitext(filename)
        filename = name[:MAX_FILENAME_LENGTH - len(ext)] + ext
    # Only allow alphanumeric, dots, dashes, underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return filename or "document.pdf"

@app.post("/upload")
async def upload(
    file: UploadFile,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a PDF document (requires authentication)"""
    try:
        # Validate filename
        if not file.filename:
            raise ValueError("Filename is required")
        
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        
        # Check file extension
        if not safe_filename.lower().endswith('.pdf'):
            raise ValueError("Please, load file in PDF format!")

        # Read file with size limit
        pdf_bytes = await file.read()
        
        # Check file size
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.0f} MB")

        if len(pdf_bytes) == 0:
            raise ValueError("Loaded file is empty!")
        
        # Validate PDF magic bytes (PDF files start with %PDF)
        if not pdf_bytes.startswith(b'%PDF'):
            raise ValueError("Invalid PDF file format")
        
    except ValueError as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "details": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "details": "An error occurred during file upload"}
        )

    try:
        doc_id, path = save_pdf(pdf_bytes)

        collection_name = f"doc_{doc_id}"
        vectordb = get_chroma(collection_name)

        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        vectordb.add_documents(chunks)
        vectordb.persist()

        db_doc = Document(
            id=doc_id,
            filename=safe_filename,
            collection_name=collection_name,
            user_id=current_user.id
        )
        db.add(db_doc)
        db.commit()

        return {"doc_id": doc_id, "status": "Success!"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "details": "Failed to process PDF file"}
        )


@app.post("/ask", response_class=JSONResponse)
async def ask(
    doc_id: str = Query(..., max_length=MAX_DOC_ID_LENGTH),
    question: str = Query(..., max_length=MAX_QUESTION_LENGTH),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Ask a question about a document (requires authentication and ownership)"""
    # Validate and sanitize inputs
    if not doc_id or not isinstance(doc_id, str):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "details": "Document ID is required"}
        )
    
    doc_id = doc_id.strip()
    if len(doc_id) > MAX_DOC_ID_LENGTH:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "details": "Document ID is too long"}
        )
    
    if not question or not isinstance(question, str):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "details": "Question is required"}
        )
    
    question = question.strip()
    if len(question) > MAX_QUESTION_LENGTH:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "details": "Question is too long"}
        )
    
    if not question:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "details": "Question is empty!"}
        )
    
    # Validate doc_id format (should be UUID)
    if not re.match(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', doc_id, re.IGNORECASE):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "details": "Invalid document ID format"}
        )
    
    # Check document ownership
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == current_user.id
    ).first()

    if not doc:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "error", "details": "Document not found or access denied!"}
        )

    try:
        vectordb = get_chroma(doc.collection_name)
        answer = answer_question(vectordb, question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "details": "Failed to process question"}
        )


@app.get("/documents", response_class=JSONResponse)
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all documents for the current user"""
    documents = db.query(Document).filter(Document.user_id == current_user.id).all()
    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "collection_name": doc.collection_name
            }
            for doc in documents
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)