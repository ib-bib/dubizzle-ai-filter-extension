import json
import os
from google import genai
from pydantic import BaseModel, Field

import modal

class KnowledgeExtraction(BaseModel):
    needs_expansion: bool = Field(description="True if the prompt asks to include or exclude a broad category of items/brands (e.g. 'chinese brands', 'american cars', 'luxury watches') that requires knowing specific names.")
    category: str = Field(description="The generic category name to expand, e.g. 'Chinese car brands'. Empty if none.")

def get_knowledge(category: str) -> str:
    try:
        # Connects to a persistent cloud dictionary
        cache = modal.Dict.from_name("dubizzle-knowledge-cache", create_if_missing=True)
    except Exception as e:
        print("Failed to connect to modal.Dict:", e)
        cache = {} # Fallback to ephemeral dict if running locally without Modal

    if category in cache:
        return cache[category]
        
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return ""
        
    client = genai.Client(api_key=api_key)
    prompt = f"The user is filtering classifieds. They mentioned the category '{category}'. Please search the web and provide an exhaustive, comma-separated list of popular brands/entities that fall under this category. Just return the comma-separated list and nothing else."
    
    from google.genai import types
    
    try:
        response = client.chats.create(
            model=os.environ.get("PRIMARY_MODEL", "gemini-2.0-flash"),
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}]
            )
        ).send_message(prompt)
        
        result = response.text.strip()
        print(f"Knowledge Manager: Retrieved list for '{category}': {result}", flush=True)
        cache[category] = result
        return result
    except Exception as e:
        print(f"Error retrieving knowledge for {category}: {e}")
        return ""

def extract_and_expand_knowledge(user_prompt: str) -> str:
    if not user_prompt.strip():
        return ""
        
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return ""
        
    client = genai.Client(api_key=api_key)
    
    schema_dict = KnowledgeExtraction.model_json_schema()
    # Gemini Developer API forbids additionalProperties in the schema
    def remove_unsupported_keys(d):
        if isinstance(d, dict):
            if "additionalProperties" in d:
                del d["additionalProperties"]
            if "title" in d:
                del d["title"]
            for k, v in list(d.items()):
                remove_unsupported_keys(v)
        elif isinstance(d, list):
            for i in d:
                remove_unsupported_keys(i)
    remove_unsupported_keys(schema_dict)
    
    try:
        response = client.chats.create(
            model=os.environ.get("PRIMARY_MODEL", "gemini-2.0-flash"),
            config={
                'response_mime_type': 'application/json',
                'response_schema': schema_dict,
            }
        ).send_message(f"Analyze this prompt: '{user_prompt}'. Does it mention a generic category that needs to be expanded into a list of specific brands? Ensure the category is specific enough for a web search (e.g. if they say 'no chinese', output 'chinese car brands' or whatever domain is implied).")
        
        data = json.loads(response.text)
        if data.get("needs_expansion") and data.get("category"):
            category = data["category"]
            print(f"Knowledge Manager: Extracted category '{category}' for expansion.", flush=True)
            list_of_brands = get_knowledge(category)
            if list_of_brands:
                return f"KNOWLEDGE BASE - The following brands belong to '{category}': {list_of_brands}"
    except Exception as e:
        print("Knowledge extraction failed:", e)
        
    return ""
