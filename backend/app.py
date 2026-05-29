from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import ( AutoTokenizer, AutoModelForCausalLM ) 

from peft import PeftModel 
import torch
# ======================================
# IMPORT YOUR EXISTING FILES
# ======================================

from upload import router as upload_router
from graph import app_graph

# ======================================
# FASTAPI APP
# ======================================

app = FastAPI()

# Upload router
app.include_router(upload_router)

# ======================================
# CORS
# ======================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# REQUEST MODEL
# ======================================

class ChatRequest(BaseModel):
    message: str

# ======================================
# LOAD FINETUNED MODEL
# ======================================

adapter_path = "../tinyllama-ssn"

base_model = AutoModelForCausalLM.from_pretrained( "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto", torch_dtype=torch.float16 )

# ====================================== # LOAD TOKENIZER # ====================================== 
tokenizer = AutoTokenizer.from_pretrained( "TinyLlama/TinyLlama-1.1B-Chat-v1.0" )

print("Loading Fine-Tuned TinyLlama...")


ft_model = PeftModel.from_pretrained( base_model, adapter_path )

print("Fine-tuned LoRA model loaded successfully")

# ======================================
# FINETUNED MODEL FUNCTION
# ======================================

def generate_ft_answer(question):

    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(ft_model.device)

    outputs = ft_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove prompt text
    answer = answer.split("Answer:")[-1].strip()

    return answer

# ======================================
# HOME ROUTE
# ======================================

@app.get("/")
def home():

    return {
        "message": "SSN AI Assistant Backend Running"
    }

# ======================================
# CHAT API
# ======================================

@app.post("/chat")
def chat(request: ChatRequest):

    user_question = request.message

    # ==================================
    # RAG RESPONSE
    # ==================================

    rag_result = app_graph.invoke({
        "question": user_question
    })

    rag_answer = rag_result["answer"]

    # ==================================
    # FINETUNED MODEL RESPONSE
    # ==================================

    ft_answer = generate_ft_answer(
        user_question
    )

    # ==================================
    # RETURN BOTH
    # ==================================

    return {

        "rag_response": rag_answer,

        "finetuned_response": ft_answer
    }

