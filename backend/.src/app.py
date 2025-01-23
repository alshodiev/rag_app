from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.rag import get_answer_and_docs
from src.qdrant import upload_website_to_collection

app = FastAPI(
    title = "RAG with FastAPI",
    description= "This is a simple example of RAG with FastAPI",
    version = "0.1"
)

@app.get("/chat")
def chat (message: str):
    response = get_answer_and_docs(message)
    response_content = {
        "question": message,
        "answer": response["answer"],
        "documents": [doc.dict() for doc in response["context"]]
    }

    return JSONResponse(content = response_content, status_code = 200)

@app.post("/indexing", description= "Index a website through this endpoint")
def indexing(url: str):
    try:
        response = upload_website_to_collection(url)
        return JSONResponse(content = {"message": response}, status_code = 200)
    except Exception as e:
        return JSONResponse(content = {"message": str(e)}, status_code = 400)
    