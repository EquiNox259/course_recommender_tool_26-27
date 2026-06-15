import os
from google import genai
from google.genai import types

class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")        
        if not api_key:
            api_key = "AQ.Ab8RN6L_0QNUdlNVC014BHKiUj0doM5__hna1STNu2FzAquRew"
            
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
    
    def rephrase_and_extract_intent(self, raw_query: str) -> str:
        """
        Transforms a conversational user query into a clean, dense string of 
        academic concepts, topics, and synonyms optimized for vector and keyword search.
        """
        system_prompt = (
            "You are an expert academic search query optimizer. Your job is to take a student's "
            "informal interest query and translate it into a space-separated string of core "
            "academic topics, subjects, technical terms, and synonyms. "
            "Rules:\n"
            "1. Remove all conversational filler (e.g., 'i want to learn', 'chill class', 'good professor', 'looking for').\n"
            "2. Expand any common tech/engineering abbreviations if present (e.g., 'ML' -> 'Machine Learning', 'AI' -> 'Artificial Intelligence').\n"
            "3. Output ONLY the clean, search-optimized keywords separated by spaces. Do not include introductory text, explanations, or quotes."
        )
        
        try:
            # Replace this with your actual Gemini API call syntax
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", 
                config={"system_instruction": system_prompt},
                contents=raw_query
            )
            optimized_query = response.text.strip()
            return optimized_query if optimized_query else raw_query
        except Exception as e:
            print(f"[LLM REPHRASER ERROR] Fallback to raw query due to: {e}")
            return raw_query