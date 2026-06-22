import os
from urllib import response
from google import genai
from google.genai import types
import json

class LLMService:
    def __init__(self):
    #    api_key = "AIzaSyCvUyZtHlfxBvWN52wTc3ESZ2H7sZSE648"
        api_key = "AQ.Ab8RN6LmuBRrdgMLYvxWcR5w5eCsiGUqgelZXcSsHl13alHCdQ"
        # Pass the key explicitly into the Client constructor
        self.client = genai.Client(api_key=api_key)
    def filter_courses(self, search_intent: str, candidate_courses: list) -> str:
        """
        Takes a student's search intent and a list of candidate course dicts,
        uses Gemini to strip out contextually irrelevant options, and returns a string.
        """
        # Formulate course text block for context
        course_string = "\n".join([
            f"- {c['code']}: {c['name']}" 
            for c in candidate_courses
        ])
        
        system_instruction = (
            "You are an academic advisor backend filter. Your job is to look at a student's search intent "
            "and a list of candidate courses. Remove courses that are contextually completely incorrect "
            "(e.g., if they ask for Btech engineering core, remove Fintech management courses). "
            "Return a clean JSON object containing an array under the key 'valid_course_codes'."
        )
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                f"Candidate Courses:\n{course_string}", 
                f"Student Intent: {search_intent}"
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                # Crucial parameter: Forces the model to return syntactically flawless JSON
                response_mime_type="application/json" 
            )
        )           
        return response.text
    
    def rephrase_and_extract_intent(self, raw_query: str) -> dict:
        """
        Validates the query for academic intent and extracts structured 
        n-gram keywords in a single, unified LLM API call.
        """
        system_prompt = (
            "You are an expert academic search query optimizer and safety gatekeeper. "
            "Your job is to analyze a student's informal query, validate it, and break it down "
            "into precise academic search concepts.\n\n"
            
            "STEP 1: VALIDATE THE INPUT\n"
            "Determine if the query contains any genuine intent to find academic courses, topics, fields, or skills.\n"
            "Mark 'is_valid' as false if the query is total gibberish/keyboard smashes ('asdfg'), profanity, "
            "or completely off-topic conversational noise ('tell me a joke'). Otherwise, mark it true.\n\n"
            
            "STEP 2: EXTRACT KEYPHRASES (Only if 'is_valid' is true)\n"
            "Extract phrases naturally without splitting compound concepts into single words. "
            "Keep them together as natural bigrams or trigrams (e.g., 'machine learning', 'computer vision'). "
            "Expand common abbreviations (e.g., 'ML' -> 'machine learning', 'AI' -> 'artificial intelligence').\n\n"
            
            "Categorize into three lists of strings:\n"
            "1. combined: All extracted concepts, fields, and structural modifiers.\n"
            "2. primary: Core academic subjects, fields, or technical terms.\n"
            "3. secondary: Modifiers, skill levels, or structural types (e.g., 'fundamentals', 'advanced', 'lab').\n\n"
            
            "CRITICAL: You must output ONLY a valid JSON object matching the examples below. "
            "Do not include markdown wrappers or conversational intro text."
            
            "\n\nExample Output (Valid Input):"
            "{\n"
            "  \"is_valid\": true,\n"
            "  \"reject_reason\": \"\",\n"
            "  \"combined\": [\"machine learning\", \"artificial intelligence\", \"advanced\"],\n"
            "  \"primary\": [\"machine learning\", \"artificial intelligence\"],\n"
            "  \"secondary\": [\"advanced\"]\n"
            "}"
            
            "\n\nExample Output (Garbage Input):"
            "{\n"
            "  \"is_valid\": false,\n"
            "  \"reject_reason\": \"Query appears to be meaningless keyboard gibberish.\",\n"
            "  \"combined\": [],\n"
            "  \"primary\": [],\n"
            "  \"secondary\": []\n"
            "}"
        )
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=raw_query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.0  # Kept low for deterministic safety sorting
                )
            )
            
            result = json.loads(response.text.strip())
            
            # Defensive structural fallbacks to ensure keys exist
            if "is_valid" not in result:
                result["is_valid"] = True
                
            for key in ["combined", "primary", "secondary"]:
                if key not in result or not isinstance(result[key], list):
                    result[key] = []
                    
            return result

        except Exception as e:
            print(f"[LLM COMBINED PIPELINE ERROR] Fallback due to: {e}")
            clean_raw = [w.strip() for w in raw_query.lower().split() if w.strip()]
            return {
                "is_valid": True,
                "reject_reason": "",
                "combined": clean_raw,
                "primary": clean_raw,
                "secondary": []
            }
