import requests
from io import BytesIO
from PIL import Image
import logging
import os
from google import genai

logger = logging.getLogger(__name__)

def check_visual_feature(image_urls: list, positive_prompt: str, negative_prompt: str = "an ordinary object", user_query: str = "") -> bool:
    """
    Downloads up to 6 images and uses Gemini to decide if the positive_prompt describes the image
    better than the negative_prompt.
    Fail-safe: Returns True if there's an error so we don't accidentally hide a good listing.
    """
    if not image_urls:
        return True # Fail-safe
        
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("No Gemini API key found for vision. Bypassing vision check.")
        return True # Fail-safe
        
    try:
        client = genai.Client(api_key=api_key)
        
        # Take up to the first 10 images to give the AI enough context (e.g. side, back, front of car)
        images_data = []
        for url in image_urls[:10]:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    images_data.append(image)
            except Exception as inner_e:
                logger.warning(f"Failed to fetch image {url}: {inner_e}")
                continue
                
        if not images_data:
            return True # Fail-safe
            
        prompt = (
            f"You are an expert AI visual filter for a classifieds website.\n"
            f"The user has specified this visual preference: '{user_query}'\n\n"
            f"I have provided up to 10 images of a single listing. Your job is to decide whether to KEEP or REJECT this listing based on their preference.\n\n"
            f"CRITICAL RULES FOR AI LOGIC:\n"
            f"1. BE VIBES-BASED & LOOSE: Do not be pedantic! If the user asks for 'open windows or open blinds', they want a bright, open, airy feel with lots of glass. Accept large glass patio doors, open balconies, or massive transparent glass walls as satisfying the prompt! Don't reject just because it's technically a 'door' instead of a 'window'.\n"
            f"2. DEFAULT TO KEEP: Only REJECT if the images provide absolute, undeniable visual proof that the item is exactly what the user hates, or if it has the complete opposite vibe (e.g. a dark, closed-off dungeon when they want open windows).\n"
            f"3. PRESUMPTION OF INNOCENCE: If you are even slightly unsure, answer KEEP.\n\n"
            f"INSTRUCTIONS:\n"
            f"Step 1: Briefly describe what you see in the images relevant to the user's request (1-2 sentences max).\n"
            f"Step 2: On a new line, write EXACTLY ONE WORD: 'KEEP' or 'REJECT'."
        )
        
        contents = images_data + [prompt]
        
        response = client.models.generate_content(
            model=os.environ.get("VISION_MODEL", "gemini-2.0-flash"),
            contents=contents
        )
        
        answer_text = response.text.strip().upper()
        # Parse the final decision from the Chain of Thought
        lines = [line.strip() for line in answer_text.split('\n') if line.strip()]
        final_word = lines[-1] if lines else ""
        
        # Safe fallback: if it doesn't explicitly say REJECT in the final line, we keep it.
        return "REJECT" not in final_word
        
    except Exception as e:
        logger.error(f"Error running Gemini vision model: {e}")
        return True # Fail-safe: don't hide the listing on fatal error

