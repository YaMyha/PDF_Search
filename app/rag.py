import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables at module level
load_dotenv()

def get_api_key():
    """Get OpenAI API key from environment variable"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your .env file or as an environment variable."
        )
    return api_key

def build_rag(pdf_file):
    # Validate PDF file is not empty
    if not pdf_file or len(pdf_file) == 0:
        raise ValueError("PDF file is empty or invalid")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Validate that PDF was loaded successfully
        if not docs:
            raise ValueError("No documents could be loaded from the PDF. The file might be empty or corrupted.")
        
        # Check if any document has content
        if all(not doc.page_content.strip() for doc in docs):
            raise ValueError("PDF contains no extractable text content. The file might be an image-only PDF or corrupted.")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        
        if not chunks:
            raise ValueError("No valid text chunks could be created from the PDF. Please ensure the PDF contains text.")
        
        api_key = get_api_key()
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectordb = Chroma.from_documents(chunks, embedding=embeddings)

        return vectordb
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def answer_question(vectordb, question):
    api_key = get_api_key()
    llm = ChatOpenAI(model="gpt-5-nano", api_key=api_key)
    retriever = vectordb.as_retriever()

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    response = llm.invoke(
        f"Контекст:\n{context}\n\n Вопрос: {question}\n\n Ответь максимально конкретно."
    )
    print(response.content)
    return response.content
