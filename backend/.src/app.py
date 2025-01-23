from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title = "RAG with FastAPI",
    description= "This is a simple example of RAG with FastAPI",
    version = "0.1"
)

@app.get("/chat")
def chat (message: str):
    return JSONResponse(content = {"Your Message": message}, status_code = 200)
