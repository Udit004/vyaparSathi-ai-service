from fastapi import FastAPI
from app.llm import get_llm
from app.schemas import ChatRequest, ChatResponse

app = FastAPI(title="VyaparSathi AI Service")

llm = get_llm()  # initialize once


@app.get("/")
async def root():
    return {"status": "AI service running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response = llm.invoke(request.message)
    return ChatResponse(response=response.content)