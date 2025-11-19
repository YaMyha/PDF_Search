import uvicorn
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import SessionLocal, Document
from app.rag import build_rag, answer_question, save_pdf, get_chroma

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(file: UploadFile):
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

    db = SessionLocal()
    db_doc = Document(id=doc_id, filename=file.filename, collection_name=collection_name)
    db.add(db_doc)
    db.commit()

    return {"doc_id": doc_id, "status": "Success!"}


@app.post("/ask", response_class=HTMLResponse)
async def ask(doc_id: str, question: str):
    db = SessionLocal()
    doc = db.query(Document).filter(Document.id == doc_id).first()

    if not doc:
        return {"status": "error", "details": "Document not found!"}

    if not question:
        return {"status": "error", "details": "Question is empty!"}

    vectordb = get_chroma(doc.collection_name)
    answer = answer_question(vectordb, question)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)