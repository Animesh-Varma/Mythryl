from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from contextlib import asynccontextmanager

from core import (
    load_resources,
    find_similar_responses,
    generate_llm_prompt,
    get_llm_response,
    add_message_to_database,
    verify_persona_name,
)

# --- Global State ---
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all necessary resources when the API starts and clean up on shutdown."""
    print("Loading resources for the API...")
    index, df, model = load_resources()
    if index is None or df is None or model is None:
        raise RuntimeError("Failed to load resources. Make sure setup.py has been run.")
    
    SENDER_NAME_PATH = "temp/sender_name.txt"
    if os.path.exists(SENDER_NAME_PATH):
        with open(SENDER_NAME_PATH, "r", encoding="utf-8") as f:
            sender_name = f.read().strip()
    else:
        # Fallback if sender_name.txt is not found
        sender_name = "User"
        print("Warning: sender_name.txt not found. Using default sender name 'User'.")

    state['index'] = index
    state['df'] = df
    state['model'] = model
    state['sender_name'] = sender_name
    print("Resources loaded successfully.")
    
    yield
    
    # Clean up the resources
    state.clear()
    print("Resources cleaned up.")

app = FastAPI(lifespan=lifespan)

# --- Data Models ---
class ChatRequest(BaseModel):
    persona: str
    message: str
    service: str = 'gemini'  # or 'ollama'
    conversation_history: list[str] = []

class ChatResponse(BaseModel):
    response: str

class PersonasResponse(BaseModel):
    personas: list[str]

class AddMessageRequest(BaseModel):
    persona: str
    prompt: str
    response: str

class VerifyPersonaRequest(BaseModel):
    persona: str

class VerifyPersonaResponse(BaseModel):
    status: str
    persona: str | None = None
    confidence: float | None = None

@app.get("/")
def read_root():
    return {"message": "Mythryl API is running."}

@app.get("/personas", response_model=PersonasResponse)
async def get_personas():
    """Returns a list of available personas."""
    try:
        df = state['df']
        personas = df['persona'].unique().tolist()
        return PersonasResponse(personas=personas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_persona(request: ChatRequest):
    """Handles a chat request with a specific persona."""
    try:
        # Retrieve resources from state
        index = state['index']
        df = state['df']
        model = state['model']
        sender_name = state['sender_name']

        # Find similar responses
        similar_responses = find_similar_responses(
            query=request.message,
            index=index,
            df=df,
            model=model,
            persona=request.persona,
            k=5  # You can make this configurable
        )

        # Generate the prompt for the LLM
        prompt_for_llm = generate_llm_prompt(
            query=request.message,
            conversation_history=request.conversation_history,
            similar_responses=similar_responses,
            persona=request.persona,
            sender_name=sender_name
        )

        # Get the response from the LLM
        bot_response = get_llm_response(
            service_choice=request.service,
            prompt_for_llm=prompt_for_llm
        )

        return ChatResponse(response=bot_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_message")
async def add_message(request: AddMessageRequest):
    """Adds a new message to the vector database."""
    try:
        df, index = add_message_to_database(
            df=state['df'],
            index=state['index'],
            model=state['model'],
            persona=request.persona,
            prompt=request.prompt,
            response=request.response
        )
        # Update the state
        state['df'] = df
        state['index'] = index
        return {"message": "Message added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify_persona", response_model=VerifyPersonaResponse)
async def verify_persona(request: VerifyPersonaRequest):
    """Verifies a persona name, suggesting the closest match if not found."""
    try:
        result = verify_persona_name(state['df'], request.persona)
        return VerifyPersonaResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=50507)
