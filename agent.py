import json
from typing import TypedDict, Annotated, List, Dict, Any
from pydantic import BaseModel, Field, TypeAdapter
import os
from google import genai
from google.genai import errors

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from vision import check_visual_feature

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    items: List[Dict[str, Any]]
    keep_ids: List[str]
    use_vision: bool

class FilterResult(BaseModel):
    text_keep_ids: List[str] = Field(description="IDs of listings that match the non-visual text constraints. If there are no text constraints, return all IDs.")
    requires_visual_inspection: bool = Field(description="True if the user query contains a visual constraint (e.g. 'kitchen island', 'angular headlights', 'red color').")
    visual_feature_positive: str = Field(description="The specific visual feature to look for (e.g. 'a kitchen island', 'angular headlights'). Leave empty if none.", default="")
    visual_feature_negative: str = Field(description="What to contrast it against (e.g. 'an ordinary kitchen without an island', 'round headlights'). Leave empty if none.", default="")

from tenacity import retry, wait_exponential, stop_after_attempt

# Initialize native Google GenAI client
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

FALLBACK_MODELS = [
    os.environ.get("PRIMARY_MODEL", "gemini-2.0-flash")
]

@retry(wait=wait_exponential(multiplier=1, min=1, max=3), stop=stop_after_attempt(2))
def generate_filter_content(contents, schema_type):
    schema_dict = schema_type.model_json_schema()
    
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

    last_error = None
    
    # 1. Try Native Google GenAI
    for model_name in FALLBACK_MODELS:
        try:
            chat = client.chats.create(
                model=model_name,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': schema_dict,
                }
            )
            return chat.send_message(contents)
        except errors.APIError as e:
            code = getattr(e, 'code', None)
            if code in (401, 403):
                raise Exception(f"Authentication error (401/403): API Key is invalid or missing permissions.")
            elif code == 400:
                raise Exception(f"Bad request (400): The prompt or schema was invalid. {e.message}")
            elif code == 404:
                print(f"Warning: Model {model_name} not found (404). Falling back...", flush=True)
                if last_error is None: last_error = e
                continue
            elif code in (429, 503):
                print(f"Warning: Model {model_name} overloaded ({code}). Falling back...", flush=True)
                last_error = e
                continue
            else:
                if last_error is None: last_error = e
                continue
        except Exception as e:
            if last_error is None: last_error = e
            continue
            
    # 2. Universal Fallback System (Groq, OpenRouter, OpenAI)
    # If we reached here, Google failed.
    import openai
    import json
    
    fallback_configs = [
        {
            "name": "OpenAI",
            "key_env": "OPENAI_API_KEY",
            "base_url": None, # default OpenAI URL
            "models": [os.environ.get("FALLBACK_OPENAI_MODEL", "gpt-4o-mini")]
        },
        {
            "name": "OpenRouter",
            "key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "models": [os.environ.get("FALLBACK_OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")]
        },
        {
            "name": "Groq",
            "key_env": "GROQ_API_KEY",
            "base_url": "https://api.groq.com/openai/v1",
            "models": [os.environ.get("FALLBACK_GROQ_MODEL", "llama-3.3-70b-versatile")]
        }
    ]
    
    for config in fallback_configs:
        api_key = os.environ.get(config["key_env"])
        if not api_key:
            continue
            
        print(f"Attempting fallback to {config['name']}...", flush=True)
        try:
            o_client = openai.OpenAI(api_key=api_key, base_url=config["base_url"])
            for f_model in config["models"]:
                try:
                    # Instruct the model to return JSON matching the schema
                    sys_prompt = f"You are a structured data extractor. You MUST return ONLY valid JSON matching this exact schema: {json.dumps(schema_dict)}"
                    response = o_client.chat.completions.create(
                        model=f_model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": contents}
                        ],
                        response_format={"type": "json_object"}
                    )
                    
                    class FakeResponse:
                        def __init__(self, text):
                            self.text = text
                    return FakeResponse(response.choices[0].message.content)
                except Exception as inner_e:
                    print(f"  -> Model {f_model} failed: {inner_e}", flush=True)
                    continue
        except Exception as provider_e:
            print(f"Provider {config['name']} completely failed: {provider_e}", flush=True)
            
    raise last_error or Exception("All fallback providers completely failed.")

def filter_node(state: AgentState):
    user_messages = state['messages']
    items = state.get('items', [])
    
    system_prompt = (
        "You are an AI assistant that acts as a filter for a classifieds website (motors & properties).\n"
        "You are operating at Step 3 of the pipeline. The items have ALREADY passed native website filters and basic keyword filters.\n"
        "The user will provide a query describing what they are looking for, or what they want to EXCLUDE.\n"
        "You will be given a JSON list of items currently on the page.\n"
        "Step 3: TEXT FILTERING. Evaluate each item against the text constraints in the DOM. Use your broad world knowledge to categorize items! Pay CLOSE attention to negative constraints (e.g. 'remove', 'no', 'exclude'). For example, if the user says 'no american cars', you MUST use your knowledge to exclude ANY car from an American brand. If the user says 'no chinese', exclude Chinese brands. For properties, if they say 'unfurnished', exclude items that say 'furnished'. Return the IDs of items that should be KEPT.\n"
        "Step 4: VISUAL FILTERING. Determine if the user query has a STRICTLY visual constraint that CANNOT be resolved from the text (e.g. 'red color', 'has a spoiler', 'damaged bumper', 'swimming pool in picture', 'kitchen island', 'bulbous headlights vs angular trim').\n"
        "DO NOT use visual inspection for country of origin, brand, model, manufacturer, or basic text properties (like 'maid room' or 'balcony' IF they are typically in the text). However, DO use vision for specific architectural details, interior design elements (like kitchen islands), or specific visual trims/shapes of cars.\n"
        "If and ONLY if there is a purely visual constraint AND vision is enabled, set requires_visual_inspection to True and provide a positive and negative prompt for an image classifier (CLIP).\n"
    )

    use_vision = state.get('use_vision', False)
    vision_instruction = (
        "VISION IS CURRENTLY ENABLED. You may use Step 4." if use_vision else 
        "CRITICAL: VISION IS CURRENTLY DISABLED. You MUST set requires_visual_inspection to False. If the user asks for a visual feature (like angular headlights or a kitchen island) and it is NOT explicitly written in the text, you MUST ignore the visual constraint and KEEP the item. It is the user's responsibility to manually verify visual traits since they turned vision off."
    )
    
    system_prompt += f"\n\n{vision_instruction}\n"

    items_text = json.dumps(items, indent=2)
    user_prompt = user_messages[-1].content if user_messages else ""
    
    from knowledge_manager import extract_and_expand_knowledge
    knowledge_context = extract_and_expand_knowledge(user_prompt)
    if knowledge_context:
        system_prompt += f"\n\n{knowledge_context}\n"
    
    # Construct the message content for the Native SDK
    contents = f"System Instructions:\n{system_prompt}\n\nListings to Evaluate:\n{items_text}\n\nUser Query:\n{user_prompt}"
    
    # Native structured output call (bypasses LangChain's AFC warnings)
    response = generate_filter_content(contents, FilterResult)
    
    # Parse the structured JSON response
    result_data = json.loads(response.text)
    filter_result = FilterResult(**result_data)
    
    final_keep_ids = []
    
    # Text filter baseline
    text_passed_items = [item for item in items if item.get('id') in filter_result.text_keep_ids]
    
    if state.get('use_vision') and filter_result.requires_visual_inspection and filter_result.visual_feature_positive:
        # Pass remaining items through the local CLIP vision model
        for item in text_passed_items:
            img_urls = item.get('image_urls', [])
            if not img_urls:
                continue 
                
            passed = check_visual_feature(
                image_urls=img_urls, 
                positive_prompt=filter_result.visual_feature_positive, 
                negative_prompt=filter_result.visual_feature_negative or "an ordinary object",
                user_query=user_prompt
            )
            
            if passed:
                final_keep_ids.append(item['id'])
    else:
        # No visual inspection needed
        final_keep_ids = filter_result.text_keep_ids
    
    return {"keep_ids": final_keep_ids}

workflow = StateGraph(AgentState)
workflow.add_node('agent', filter_node)
workflow.add_edge(START, 'agent')

app_graph = workflow.compile()