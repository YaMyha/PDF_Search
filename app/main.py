import uvicorn
from datetime import timedelta
from fastapi import FastAPI, UploadFile, Form, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.models import SessionLocal, Document, User
from app.rag import answer_question, save_pdf, get_chroma
from app.auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_user, get_db, ACCESS_TOKEN_EXPIRE_MINUTES
)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")


# Pydantic models for request/response
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user"""
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
    
    # Validate password strength
    if len(user_data.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # Create new user
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
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticate user and return access token"""
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

@app.post("/upload")
async def upload(
    file: UploadFile,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a PDF document (requires authentication)"""
    try:
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise ValueError("Please, load file in PDF format!")

        pdf_bytes = await file.read()

        if len(pdf_bytes) == 0:
            raise ValueError("Loaded file is empty!")
    except Exception as e:
        return {"status": "error", "details": str(e)}

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
        filename=file.filename,
        collection_name=collection_name,
        user_id=current_user.id
    )
    db.add(db_doc)
    db.commit()

    return {"doc_id": doc_id, "status": "Success!"}


@app.post("/ask", response_class=JSONResponse)
async def ask(
    doc_id: str,
    question: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Ask a question about a document (requires authentication and ownership)"""
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == current_user.id
    ).first()

    if not doc:
        return {"status": "error", "details": "Document not found or access denied!"}

    if not question:
        return {"status": "error", "details": "Question is empty!"}

    vectordb = get_chroma(doc.collection_name)
    answer = answer_question(vectordb, question)
    return {"answer": answer}


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