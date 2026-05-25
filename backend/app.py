from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from upload import router as upload_router
from graph import app_graph

# =========================================
# FASTAPI APP
# =========================================

app = FastAPI()

# Upload routes
app.include_router(upload_router)

# =========================================
# ENABLE CORS
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# REQUEST MODEL
# =========================================

class ChatRequest(BaseModel):
    message: str

# =========================================
# HEALTH CHECK
# =========================================

@app.get("/")
def home():

    return {
        "message": "SSN chatbot backend running"
    }

# =========================================
# CHAT ENDPOINT
# =========================================

@app.post("/chat")
def chat(request: ChatRequest):

    result = app_graph.invoke({
        "question": request.message
    })

    return {
        "response": result["answer"]
    }