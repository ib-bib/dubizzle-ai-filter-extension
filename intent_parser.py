import json
import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import errors
from tenacity import retry, wait_exponential, stop_after_attempt

class IntentParseResult(BaseModel):
    category_path: str = Field(description="The dubizzle category path, e.g., '/motors/used-cars/', '/property-for-rent/residential/', or just '/' if unknown")
    native_filters: dict[str, str] = Field(description="A dictionary of structural URL query parameters (e.g., {'price__lte': '45000', 'price__gte': '10000', 'year__gte': '2015'}). Use generic terms if you don't know the exact dubizzle param.")
    ai_prompt: str = Field(description="The remaining qualitative or subjective part of the prompt that cannot be structurally filtered by the website (e.g., 'anything but chinese', 'looks sporty', 'has a large kitchen').")

FALLBACK_MODELS = [
    os.environ.get("PRIMARY_MODEL", "gemini-2.0-flash")
]

import re

def fast_local_parse(prompt: str) -> dict:
    prompt_lower = prompt.lower()
    native_filters = {}
    
    # 1. Price
    price_lte_match = re.search(r'(?:less than|under|<|max|below)\s*(\d+(?:k|000)?)\s*(?:aed|dhs)?', prompt_lower)
    if price_lte_match:
        val_str = price_lte_match.group(1).replace('k', '000')
        native_filters['price__lte'] = val_str
        prompt_lower = prompt_lower.replace(price_lte_match.group(0), '')
        
    price_gte_match = re.search(r'(?:more than|over|>|min|above)\s*(\d+(?:k|000)?)\s*(?:aed|dhs)?', prompt_lower)
    if price_gte_match:
        val_str = price_gte_match.group(1).replace('k', '000')
        native_filters['price__gte'] = val_str
        prompt_lower = prompt_lower.replace(price_gte_match.group(0), '')

    # 2. Kilometers
    km_lte_match = re.search(r'(?:less than|under|<|max|below)\s*(\d+(?:k|000)?)\s*(?:km|kilometers|mileage)', prompt_lower)
    if km_lte_match:
        val_str = km_lte_match.group(1).replace('k', '000')
        native_filters['kilometers__lte'] = val_str
        prompt_lower = prompt_lower.replace(km_lte_match.group(0), '')
        
    km_gte_match = re.search(r'(?:more than|over|>|min|above)\s*(\d+(?:k|000)?)\s*(?:km|kilometers|mileage)', prompt_lower)
    if km_gte_match:
        val_str = km_gte_match.group(1).replace('k', '000')
        native_filters['kilometers__gte'] = val_str
        prompt_lower = prompt_lower.replace(km_gte_match.group(0), '')

    # 3. Year
    year_lte_match = re.search(r'(?:before|older than|max year)\s*(\d{4})', prompt_lower)
    if year_lte_match:
        native_filters['year__lte'] = year_lte_match.group(1)
        prompt_lower = prompt_lower.replace(year_lte_match.group(0), '')
        
    year_gte_match = re.search(r'(?:after|newer than|min year|since)\s*(\d{4})', prompt_lower)
    if year_gte_match:
        native_filters['year__gte'] = year_gte_match.group(1)
        prompt_lower = prompt_lower.replace(year_gte_match.group(0), '')
        
    # Check for negations to prevent naive extraction from breaking negative constraints
    negation_words = ['no', 'remove', 'except', 'not', 'without', 'exclude', 'anything but', 'other than']
    has_negation = any(w in prompt_lower for w in negation_words)

    # 4. Specs
    if not has_negation:
        specs_map = {
            'gcc': 'GCC',
            'japanese': 'Japanese',
            'american': 'North American',
            'european': 'European'
        }
        for spec, val in specs_map.items():
            if spec in prompt_lower:
                native_filters['regional_specs'] = val
                prompt_lower = re.sub(r'\b' + spec + r'(?:\s+specs)?\b', '', prompt_lower)

    # 4. Body Type
    if not has_negation:
        body_types = ['sedan', 'suv', 'hatchback', 'coupe', 'convertible']
        for bt in body_types:
            if bt in prompt_lower:
                native_filters['body_type'] = bt.capitalize()
                prompt_lower = re.sub(r'\b' + bt + r'\b', '', prompt_lower)

    # 5. Bedrooms
    bed_match = re.search(r'(\d+)\s*(?:bedrooms|beds|bhk)', prompt_lower)
    if bed_match:
        native_filters['bedrooms'] = bed_match.group(1)
        prompt_lower = prompt_lower.replace(bed_match.group(0), '')
    elif 'studio' in prompt_lower:
        native_filters['bedrooms'] = '0'
        prompt_lower = re.sub(r'\bstudio\b', '', prompt_lower)
        
    # Clean up filler words
    filler_words = [r'\baed\b', r'\bkm\b', r'\bmileage\b', r'\bwith\b', r'\band\b', r'\bor\b', r'\bcars?\b', r'\bspecs?\b', r'\bless\b', r'\bthan\b', r'\bunder\b', r'\bbelow\b']
    for fw in filler_words:
        prompt_lower = re.sub(fw, '', prompt_lower)
        
    prompt_lower = re.sub(r'\s+', '', prompt_lower).strip()
    
    # Assign likely category based on extracted filters or original prompt
    category = "/"
    original_prompt = prompt.lower()
    if any(w in original_prompt for w in ['rent', 'lease', 'tenant']):
        category = "/property-for-rent/residential/"
    elif any(w in original_prompt for w in ['buy', 'sale', 'purchase', 'own']):
        category = "/property-for-sale/residential/"
    elif any(w in original_prompt for w in ['apartment', 'villa', 'studio', 'bhk', 'property', 'house']):
        # Default to rent if ambiguous
        category = "/property-for-rent/residential/"
    elif any(w in original_prompt for w in ['car', 'cars', 'motor', 'vehicle', 'sedan', 'suv', 'hatchback', 'coupe', 'convertible', 'kilometer', 'km', 'mileage']):
        category = "/motors/used-cars/"
    elif 'bedrooms' in native_filters:
        category = "/property-for-rent/residential/"
    elif any(k in native_filters for k in ['kilometers__lte', 'kilometers__gte', 'body_type', 'regional_specs']):
        category = "/motors/used-cars/"

    # If prompt is completely exhausted, return local parse!
    if len(prompt_lower) <= 2:
        return {
            "category_path": category,
            "native_filters": native_filters,
            "ai_prompt": ""
        }
    return None

@retry(wait=wait_exponential(multiplier=1, min=1, max=3), stop=stop_after_attempt(2))
def parse_user_intent(user_prompt: str) -> dict:
    # 1. Attempt ultra-fast local regex parsing first!
    fast_result = fast_local_parse(user_prompt)
    if fast_result is not None:
        return fast_result
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    system_instruction = (
        "You are an intelligent router for Dubizzle (a classifieds website in the UAE).\n"
        "Your job is to take a natural language search query and split it into two parts:\n"
        "1. Structural filters: Things the website can filter natively (price, year, category, mileage, bedrooms).\n"
        "2. AI filters: Subjective or negative constraints (e.g., 'no chinese cars', 'no american specs', 'needs a big balcony').\n"
        "CRITICAL RULES:\n"
        "- CATEGORY ROUTING:\n"
        "  * Cars/vehicles/specs/models/brands -> '/motors/used-cars/'\n"
        "  * Renting properties (apartments, villas, rent) -> '/property-for-rent/residential/'\n"
        "  * Buying properties (sale, buy, purchase) -> '/property-for-sale/residential/'\n"
        "- VALID NATIVE FILTER KEYS: Use ONLY valid keys like 'price__lte', 'price__gte', 'year__lte', 'year__gte', 'kilometers__lte', 'kilometers__gte', 'bedrooms', 'bathrooms', 'body_type', 'regional_specs'. NEVER invent keys like 'mileage__lte' or 'color'. Omit keys entirely if the user says 'any' (e.g., do NOT output 'regional_specs': 'any').\n"
        "- NEGATIONS: NEVER put negative constraints (e.g., 'no american specs', 'except chinese', 'unfurnished') into native_filters! Native filters can only INCLUDE, not exclude. Put ALL negative/exclusion constraints strictly into ai_prompt.\n"
        "Output a JSON object separating these concerns."
    )
    
    contents = f"System Instructions:\n{system_instruction}\n\nUser Query:\n{user_prompt}"
    
    schema_dict = IntentParseResult.model_json_schema()
    
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
    for model_name in FALLBACK_MODELS:
        try:
            chat = client.chats.create(
                model=model_name,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': schema_dict,
                }
            )
            response = chat.send_message(contents)
            return json.loads(response.text)
        except errors.APIError as e:
            code = getattr(e, 'code', None)
            if code in (401, 403):
                raise Exception(f"Authentication error (401/403): API Key is invalid or missing permissions.")
            elif code == 400:
                raise Exception(f"Bad request (400): The prompt or schema was invalid. {e.message}")
            elif code == 404:
                print(f"Warning: Model {model_name} not found (404). Falling back to next model...", flush=True)
                if last_error is None:
                    last_error = e
                continue
            elif code in (429, 503):
                print(f"Warning: Model {model_name} overloaded ({code}). Falling back to next model...", flush=True)
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
                    
                    return json.loads(response.choices[0].message.content)
                except Exception as inner_e:
                    print(f"  -> Model {f_model} failed: {inner_e}", flush=True)
                    continue
        except Exception as provider_e:
            print(f"Provider {config['name']} completely failed: {provider_e}", flush=True)
            
    raise last_error or Exception("All fallback providers completely failed.")
