import json
import os
from typing import cast, List, Optional, Dict, Any
import modal

# 1. Define the Modal Environment and Dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]", 
        "google-genai", 
        "pydantic", 
        "langchain-core", 
        "langgraph", 
        "Pillow",
        "python-dotenv",
        "tenacity",
        "openai"
    )
    .add_local_file("agent.py", remote_path="/root/agent.py")
    .add_local_file("vision.py", remote_path="/root/vision.py")
    .add_local_file("intent_parser.py", remote_path="/root/intent_parser.py")
    .add_local_file("usage_tracker.py", remote_path="/root/usage_tracker.py")
    .add_local_file("usage.db", remote_path="/root/usage.db")
    .add_local_file("knowledge_manager.py", remote_path="/root/knowledge_manager.py")
)

# 2. Create the Modal App
modal_app = modal.App(name="dubizzle-ai-extension-backend")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import AgentState, app_graph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import usage_tracker

load_dotenv()

app = FastAPI(title="Dubizzle Agent API")

# 3. Mount the FastAPI app to a Modal GPU function
@modal_app.function(
    image=image, 
    gpu="T4", 
    secrets=[modal.Secret.from_name("custom-secret")] # INJECTS your cloud secrets!
)
@modal.asgi_app()
def fastapi_app():
    return app

# Add CORS Middleware to allow requests from the extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ListingItem(BaseModel):
    id: str
    title: str
    price: str
    url: str
    image_urls: List[str] = []
    description: Optional[str] = None

class QueryRequest(BaseModel):
    prompt: str
    items: List[ListingItem]
    auth_token: Optional[str] = None
    use_vision: bool = False
    is_new_prompt: bool = True

@app.post("/api/filter")
def filter_listings(request: QueryRequest):
    # Determine user_id
    user_id = request.auth_token or "anonymous_guest"

    # 1. Budget Check
    budget_status = usage_tracker.check_budget(user_id=user_id, is_vision=request.use_vision)
    if not budget_status.get("allowed", True):
        return {
            "disabled": True,
            "reason": budget_status.get("reason"),
            "reset_time": budget_status.get("reset_time")
        }
        
    # Pass the user query and items into the state
    inputs = cast(AgentState, {
        "messages": [HumanMessage(content=request.prompt)],
        "items": [item.dict() for item in request.items],
        "use_vision": request.use_vision
    })

    import tenacity
    
    try:
        final_state = app_graph.invoke(inputs)
    except tenacity.RetryError as e:
        last_exc = str(e.last_attempt.exception()) if e.last_attempt else str(e)
        return {
            "disabled": True,
            "reason": f"API Error: {last_exc}"
        }
    except Exception as e:
        error_str = str(e)
        if "Authentication error" in error_str:
            return {"disabled": True, "reason": "Invalid API Key in your Modal Secrets."}
        elif "Bad request" in error_str:
            return {"disabled": True, "reason": "Bad request sent to Gemini API."}
        elif "503" in error_str or "ServerError" in error_str or "overloaded" in error_str:
            return {"disabled": True, "reason": "Google's Gemini AI is currently experiencing an outage. Please try again in a few minutes."}
        raise e

    # 2. Estimate usage and add to tracker
    raw_input_text = request.prompt + json.dumps([item.dict() for item in request.items])
    estimated_tokens = len(raw_input_text) / 4.0
    estimated_cost_usd = estimated_tokens * (0.10 / 1_000_000) 
    
    if request.is_new_prompt:
        usage_tracker.add_usage(user_id=user_id, usd_cost=estimated_cost_usd, is_vision=request.use_vision)
        # Manually increment the stats for the response since we just added usage
        budget_status["current_requests"] = budget_status.get("current_requests", 0) + 1
        if request.use_vision:
            budget_status["current_vision"] = budget_status.get("current_vision", 0) + 1

    keep_ids = final_state.get('keep_ids', [])

    return {
        "keep_ids": keep_ids,
        "usage": {
            "current_requests": budget_status.get("current_requests", 0),
            "current_vision": budget_status.get("current_vision", 0),
            "total_limit": budget_status.get("total_limit", 15),
            "vision_limit": budget_status.get("vision_limit", 5)
        }
    }

from intent_parser import parse_user_intent

class IntentRequest(BaseModel):
    prompt: str
    auth_token: Optional[str] = None

@app.post("/api/parse_intent")
def parse_intent_endpoint(request: IntentRequest):
    user_id = request.auth_token or "anonymous_guest"
    budget_status = usage_tracker.check_budget(user_id=user_id)
    if not budget_status.get("allowed", True):
         return {
            "disabled": True,
            "reason": budget_status.get("reason")
        }
        
    import tenacity
    try:
        result = parse_user_intent(request.prompt)
    except tenacity.RetryError as e:
        last_exc = str(e.last_attempt.exception()) if e.last_attempt else str(e)
        return {
            "disabled": True,
            "reason": f"API Error: {last_exc}"
        }
    except Exception as e:
        error_str = str(e)
        if "Authentication error" in error_str:
            return {"disabled": True, "reason": "Invalid API Key in your Modal Secrets."}
        elif "Bad request" in error_str:
            return {"disabled": True, "reason": "Bad request sent to Gemini API."}
        elif "503" in error_str or "ServerError" in error_str or "overloaded" in error_str:
            return {"disabled": True, "reason": "Google's Gemini AI is currently experiencing an outage. Please try again in a few minutes."}
        raise e
    usage_tracker.add_usage(user_id=user_id, usd_cost=0.0001) # minimal cost for small prompt
    return result

@app.get("/")
def main():
    return {"message": "Hello Dubizzle Backend"}
