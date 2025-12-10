
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(tmp_path)
docs = loader.load()

# Validate that PDF was loaded successfully
if not docs:
    raise ValueError("No documents could be loaded from the PDF. The file might be empty or corrupted.")

# Check if any document has content
if all(not doc.page_content.strip() for doc in docs):
    raise ValueError("PDF contains no extractable text content. The file might be an image-only PDF or corrupted.")
