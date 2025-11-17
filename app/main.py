import uvicorn
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.rag import build_rag, answer_question

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse)
async def ask(
    request: Request,
    file: UploadFile,
    question: str = Form(...)
):
    error = None
    answer = None
    
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise ValueError("Пожалуйста, загрузите файл в формате PDF")
        
        pdf_bytes = await file.read()
        
        if len(pdf_bytes) == 0:
            raise ValueError("Загруженный файл пуст")
        
        vectordb = build_rag(pdf_bytes)
        answer = answer_question(vectordb, question)
        
    except Exception as e:
        error = str(e)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "answer": answer,
        "error": error
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)