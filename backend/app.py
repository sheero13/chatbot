from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from upload import router as upload_router
from graph import app_graph

# FastAPI app
app = FastAPI()
app.include_router(upload_router)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():

    return {
        "message": "SSN chatbot backend running"
    }

@app.post("/chat")
def chat(request: ChatRequest):

    result = app_graph.invoke({
        "question": request.message
    })

    return {
        "response": result["answer"]
    }