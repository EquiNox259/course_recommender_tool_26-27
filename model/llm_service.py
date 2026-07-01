import os
from google import genai
from google.genai import types
import json
from pydantic import BaseModel, Field

class Constraints(BaseModel):
    include_departments: list[str] = Field(default_factory=list, description="Departments whose courses should be recommended, e.g., ['EE', 'CS']")    
    exclude_departments: list[str] = Field(default_factory=list, description="Departments whose courses should not be recommended, e.g., ['EE', 'CS']")
    exclude_courses: list[str] = Field(default_factory=list, description="Specific course codes that should not be recommended, e.g., ['CS 747']")
    minor: list[str] = Field(default_factory=list, description="Minor basket(s) the user wants recommendations from")
    easy_grading: bool = Field(default=False, description="True if the user explicitly asks for easy grading, GPA booster, etc.")

class QueryOptimization(BaseModel):
    is_valid: bool = Field(description="True if query contains genuine academic intent, False otherwise")
    reject_reason: str = Field(description="Explanation of why query is invalid, or empty string if valid")
    combined: list[str] = Field(description="Every extracted concept, field, and structural modifier")
    primary: list[str] = Field(description="Core academic subjects and technical concepts only")
    secondary: list[str] = Field(description="Modifiers such as 'advanced', 'introductory', 'fundamentals', etc.")
    expanded: list[str] = Field(default_factory=list, description="Academic/career context expansion generated using Steps 3 and 4")
    constraints: Constraints = Field(default_factory=lambda: Constraints(), description="Structured constraint object")
    minor_query_type: str = Field(default="simple", description="'simple' if the query only asks for a list of minor courses with no extra filters; 'stacked' if the query asks for minor courses AND additional constraints (e.g. easy grading, topic exclusions, difficulty filters). Set to 'simple' if no minor is mentioned.")

class ValidCourses(BaseModel):
    valid_course_codes: list[str] = Field(description="List of course codes that are contextually correct and related to search intent, containing at most 20 of the most relevant courses.")

class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")        
        if not api_key:
   
            api_key = "AIzaSyCVVo74-qZs-1wjsl5ckHMUQpkAbE1izP0"
            
        if not api_key:
            self.client = None
        else:
            # Pass the key explicitly into the Client constructor
            self.client = genai.Client(api_key=api_key)

    def filter_courses(self, original_query: str, search_intent: dict, candidate_courses: list) -> str:
        """
        Takes a student's original query, search intent and a list of candidate course dicts,
        uses Gemini to strip out contextually irrelevant options, and returns a string.
        """
        if not self.client:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
            
        # Formulate course text block for context (truncate descriptions to 250 chars to save input tokens)
        course_string = "\n".join([
            f"- {c['code']}: {c['name']} (Description: {c.get('description')[:250] if isinstance(c.get('description'), str) else 'N/A'}...)" 
            for c in candidate_courses
        ])
        
        system_instruction = (
            "You are an academic advisor backend filter. Your job is to look at a student's original search query, "
            "their parsed search intent, and a list of candidate courses with their descriptions. "
            "Remove courses that are contextually completely incorrect, unrelated, or violate any exclusions/negations "
            "specified in the original search query. "
            "Select at most 20 of the most relevant courses. "
            "Return a clean JSON object containing an array under the key 'valid_course_codes'."
            "For ML related queries, unless explicity stated, do not recomend courses that are the application of ML in a very specific highly unrelated field" \
            "like Geophysics, energy etc. "
        )
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                f"Candidate Courses:\n{course_string}", 
                f"Student Original Query: {original_query}",
                f"Student Parsed Intent: {json.dumps(search_intent)}"
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ValidCourses,
                temperature=0.0,
                seed=42,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )           
        return response.text
    
    def rephrase_and_extract_intent(self, raw_query: str) -> dict:
        """
        Validates the query for academic intent and extracts structured 
        n-gram keywords in a single, unified LLM API call.
        """
        if not self.client:
            # If no client is available, raise error so fallback logic in recommender runs
            raise ValueError("LLM client not initialized due to missing API Key.")

        system_prompt = (
            "You are an expert academic search query optimizer and safety gatekeeper. "
            "Your job is to analyze a student's informal query, validate it, and convert it into structured academic search concepts for a university course recommender.\n\n"


            "STEP 1: VALIDATE THE INPUT\n"
            "Determine whether the query expresses genuine intent to search for academic courses, topics, fields, careers, research areas, or technical skills.\n"
            "Set 'is_valid' to false only if the query is meaningless keyboard gibberish (e.g. 'asdfgh'), profanity with no academic intent, or completely unrelated conversational text (e.g. 'tell me a joke'). "
            "Otherwise set 'is_valid' to true.\n\n"


            "STEP 2: EXTRACT KEYPHRASES (Only if 'is_valid' is true)\n"
            "Extract complete academic phrases. Never split multi-word concepts into individual words. "
            "For example, return 'machine learning' instead of both 'machine' and 'learning'. "
            "Likewise return 'computer vision', 'fluid mechanics', 'control systems', and 'natural language processing' as complete phrases.\n"
            "Expand common abbreviations whenever possible (e.g. 'dsa' -> 'data structures and algorithms, 'ML' -> 'machine learning', 'AI' -> 'artificial intelligence', 'NLP' -> 'natural language processing',' CV' -> 'computer vision', 'RL' -> 'reinforcement learning', 'OR' -> 'operations research' ).\n\n"


            "STEP 3: ACADEMIC CONTEXT EXPANSION\n"
            "Generate an 'expanded' list containing additional academic concepts that are strongly related to the user's goals.\n"
            "Do NOT generate synonyms.\n"
            "Instead include prerequisite subjects, supporting mathematics, foundational topics, neighboring research areas, commonly co-studied subjects, and advanced follow-up topics.\n"
            "For example:\n"
            "- 'machine learning' may expand to 'optimization', 'probability', 'statistics', 'linear algebra', 'deep learning', and 'pattern recognition'.\n"
            "- 'computer vision' may expand to 'image processing', 'deep learning', and 'geometry'.\n\n"


            "STEP 4: CAREER CONTEXT EXPANSION\n"
            "If the user mentions a profession, career goal, research domain, or industry instead of academic subjects, infer the academic topics typically required.\n"
            "For example:\n"
            "- 'consulting' -> 'operations research', 'optimization', 'statistics', 'decision analysis', 'economics', 'business strategy', 'data analytics'.\n"
            "- 'quantitative finance' -> 'probability', 'stochastic processes', 'optimization', 'time series', 'financial mathematics'.\n"
            "- 'robotics engineer' -> 'control systems', 'embedded systems', 'computer vision', 'signal processing', 'motion planning'.\n"
            "Limit the expanded list to at most 10 phrases. "
            "Do not invent unrelated concepts.\n\n"
           
            "STEP 5: CONSTRAINT IDENTIFICATION\n"
            "Identify any explicit constraints or preferences specified by the user regarding the returned courses.\n"
            "Populate the 'constraints' object using only information explicitly mentioned by the user.\n"
            "Do not infer constraints that were not stated.\n\n"


            "Supported constraints:\n"
            "- include_departments: Departments whose courses should only be recommended"
            "- exclude_departments: Departments whose courses should not be recommended.\n"
            "- exclude_courses: Specific course codes that should not be recommended.\n"
            "- minor: Minor basket(s) the user wants recommendations from.\n"
            "- easy_grading: Set to true only if the user explicitly asks for easy grading, high CPI, GPA booster, light workload, easy courses, scoring courses, or similar.\n\n"
              "This includes requests for:"
            "- easy grading\n"
            "- generous grading\n"
            "- high AA percentage\n"
            "- high AA+AB percentage\n"
            "- grade-friendly courses\n"
            "- scoring courses\n"
            "- easy courses\n"
            "- GPA/CPI/CG/CGPA boosters\n"
            "- courses that are easy to get good grades in\n"
            "- courses with high grades\n"
            "- courses with lenient grading\n"
            "Infer this intent semantically even if the exact wording differs.\n"
            "Otherwise return false.\n"
           
            "STEP 6: MINOR QUERY TYPE CLASSIFICATION\n"
            "If the query mentions a minor program, set minor_query_type to:\n"
            "- 'simple': the user ONLY wants the course list for that minor — no grading, topic, or difficulty filters.\n"
            "- 'stacked': the user wants minor courses AND at least one additional constraint "
            "(e.g. easy grading, no algebra, specific topic, avoid heavy workload, grade preferences).\n"
            "If no minor is mentioned, set minor_query_type to 'simple'.\n\n"

            "Department and Minor Code Reference:\n"
            "- Centre for Machine Intelligence and Data Science: CMInDS (cminds, centre for machine intelligence, machine intelligence and data science, machine intelligence, data science, ds)\n"
            "- Computer Science and Engineering: CSE (cs, cse, computer science)\n"
            "- Electrical Engineering: EE (electrical engineering, electrical)\n"
            "- Mechanical Engineering: ME (mechanical engineering, mechanical, mech)\n"
            "- Metallurgical Engineering and Materials Science: MEMS (metallurgical engineering, metallurgical, materials science, mems)\n"
            "- Chemistry: CH (chemistry)\n"
            "- Mathematics: MATH (mathematics, math, maths)\n"
            "- Physics: PH (physics)\n"
            "- Economics: EC (economics)\n"
            "- Shailesh J. Mehta School of Management: SOM (management, som)\n"
            "- Desai Sethi School of Entrepreneurship: DSSE (entrepreneurship, ent)\n"
            "- Statistics: STATS (statistics, stats)\n"
            "- Biosciences and Bioengineering: BSBE (bio, bsbe, biosciences, bioengineering)\n"
            "- Aerospace Engineering: AE (aero, aerospace)\n"
            "- Centre for Systems and Control: SYSCON (syscon, systems and control)\n"
            "- Centre of Studies in Resources Engineering: GNR (csre, geoinformatics, gnr, resources engineering)\n"
            "- Robotics: ROBOTICS (robotics)\n"
            "- Industrial Engineering and Operations Research: IEOR (industrial engineering, operations research, ieor)\n"
            "\n"
            "When extracting department constraints, always return ONLY the department code (e.g., 'EE', 'CS', 'MM')."
           


            "Examples:\n"
            "- 'exclude electrical engineering courses' -> exclude_departments = ['EE']\n"
            "- 'don't recommend mechanical courses' -> exclude_departments = ['ME']\n"
            "- 'exclude CS 747 and EE 769' -> exclude_courses = ['CS 747', 'EE 769']\n"
            "- 'recommend courses from the AI minor' -> minor = ['Artificial Intelligence']\n"
            "- 'recommend courses from the CS minor' -> minor = ['CS']\n"
            "- 'I need an easy grading course' -> easy_grading = true\n"
            "- 'Include courses from EE department' -> include_departments = ['EE']"
            "- 'I want subjects from Mech department' -> include_departments = ['ME']"
            "- If the user specifies none of these constraints, return empty lists and false.\n\n"


            "Return the extracted information using these fields:\n"
            "1. combined: Every extracted concept, field, and structural modifier and contains the union of the elements in primary and secondary.\n"
            "2. primary: Core academic subjects and technical concepts only.\n"
            "3. secondary: Modifiers such as 'advanced', 'introductory', 'fundamentals', 'laboratory', 'project', 'seminar', etc.\n"
            "4. expanded: Academic context expansion generated using Steps 3 and 4.\n"
            "5. constraints: The structured constraint object described above.\n\n"


            "CRITICAL REQUIREMENTS:\n"
            "- If there is no mention of a concept or a career path in the query, leave primary, secondary, combined and expanded empty"
            "- Every list element must be a complete phrase, never partial words.\n"
            "- Never split compound concepts into separate tokens.\n"
            "- Do not duplicate phrases across lists unless appropriate.\n"
            "- Do not infer constraints that the user did not explicitly request.\n"
            "- Output ONLY valid JSON.\n"
            "- Do not output markdown.\n"
            "- Do not output explanations.\n\n"
            "- Only set easy_grading to true if the user explicitly expresses a preference for obtaining higher grades or easier evaluation.\n"
            "- Do not infer this preference merely because the user asks for beginner courses, introductory courses, or fundamental courses. \n"
            "- A request for introductory material does not necessarily imply easy grading.\n"
            "- If the prompt says 'courses from EE minor', include departments should be left empty, but minor should have 'EE'"
            "- If the prompt says 'courses from EE minor and CS department', include departments should only have 'CS', and minor should have 'EE'"




            "Example Output (Valid Input):\n"
            "{\n"
            "  \"is_valid\": true,\n"
            "  \"reject_reason\": \"\",\n"
            "  \"combined\": [\"machine learning\", \"computer vision\", \"advanced\"],\n"
            "  \"primary\": [\"machine learning\", \"computer vision\"],\n"
            "  \"secondary\": [\"advanced\"],\n"
            "  \"expanded\": [\n"
            "    \"optimization\",\n"
            "    \"probability\",\n"
            "    \"statistics\",\n"
            "    \"linear algebra\",\n"
            "    \"deep learning\",\n"
            "    \"neural networks\",\n"
            "    \"pattern recognition\",\n"
            "    \"image processing\"\n"
            "  ],\n"
            "  \"constraints\": {\n"
            "    \"include departments\":[]"
            "    \"exclude_departments\": [],\n"
            "    \"exclude_courses\": [],\n"
            "    \"minor\": [],\n"
            "    \"easy_grading\": false\n"
            "  }\n"
            "}\n\n"


            "Example Output (With Constraints):\n"
            "{\n"
            "  \"is_valid\": true,\n"
            "  \"reject_reason\": \"\",\n"
            "  \"combined\": [\"machine learning\"],\n"
            "  \"primary\": [\"machine learning\"],\n"
            "  \"secondary\": [\"fundamental\", \"basic\"],\n"
            "  \"expanded\": [\"optimization\", \"probability\", \"statistics\"],\n"
            "  \"constraints\": {\n"
            "    \"include_departments\":[\"CS\"]"
            "    \"exclude_departments\": [\"EE\", \"ME\"],\n"
            "    \"exclude_courses\": [\"CS 747\"],\n"
            "    \"minor\": [],\n"
            "    \"easy_grading\": true\n"
            "  }\n"
            "}\n\n"
           
            "Example Output (For minor — simple):\n"
            "{\n"
            "  \"is_valid\": false,\n"
            "  \"reject_reason\": \"\",\n"
            "  \"combined\": [],\n"
            "  \"primary\": [],\n"
            "  \"secondary\": [],\n"
            "  \"expanded\": [],\n"
            "  \"constraints\": {\n"
            "    \"include departments\":[]"
            "    \"exclude_departments\": [],\n"
            "    \"exclude_courses\": [],\n"
            "    \"minor\": [\"CS\"],\n"
            "    \"easy_grading\": false\n"
            "  },\n"
            "  \"minor_query_type\": \"simple\"\n"
            "}\n\n"

            "Example Output (For minor — stacked, e.g. 'CS minor courses with good grading and no algebra'):\n"
            "{\n"
            "  \"is_valid\": true,\n"
            "  \"reject_reason\": \"\",\n"
            "  \"combined\": [\"good grading\", \"no algebra\"],\n"
            "  \"primary\": [\"good grading\"],\n"
            "  \"secondary\": [],\n"
            "  \"expanded\": [],\n"
            "  \"constraints\": {\n"
            "    \"include_departments\": [],\n"
            "    \"exclude_departments\": [],\n"
            "    \"exclude_courses\": [],\n"
            "    \"minor\": [\"CS\"],\n"
            "    \"easy_grading\": true\n"
            "  },\n"
            "  \"minor_query_type\": \"stacked\"\n"
            "}"


            "Example Output (Invalid Input):\n"
            "{\n"
            "  \"is_valid\": false,\n"
            "  \"reject_reason\": \"Query appears to be meaningless keyboard gibberish.\",\n"
            "  \"combined\": [],\n"
            "  \"primary\": [],\n"
            "  \"secondary\": [],\n"
            "  \"expanded\": [],\n"
            "  \"constraints\": {\n"
            "    \"include departments\":[]"
            "    \"exclude_departments\": [],\n"
            "    \"exclude_courses\": [],\n"
            "    \"minor\": [],\n"
            "    \"easy_grading\": false\n"
            "  }\n"
            "}"
        )

        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=raw_query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=QueryOptimization,
                    temperature=0.0,  # Kept low for deterministic safety sorting
                    seed=42,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )

            result = json.loads(response.text.strip())
            
            # Defensive structural fallbacks to ensure keys exist
            if "is_valid" not in result:
                result["is_valid"] = True
                
            for key in ["combined", "primary", "secondary", "expanded"]:
                if key not in result or not isinstance(result[key], list):
                    result[key] = []
                    
            if "constraints" not in result or not isinstance(result["constraints"], dict):
                result["constraints"] = {
                    "exclude_departments": [],
                    "exclude_courses": [],
                    "minor": [],
                    "easy_grading": False
                }
            else:
                c_dict = result["constraints"]
                for k in ["exclude_departments", "exclude_courses", "minor"]:
                    if k not in c_dict or not isinstance(c_dict[k], list):
                        c_dict[k] = []
                if "easy_grading" not in c_dict:
                    c_dict["easy_grading"] = False

            if "minor_query_type" not in result or result["minor_query_type"] not in ("simple", "stacked"):
                result["minor_query_type"] = "simple"

            return result
        
        except Exception as e:
            print(f"[LLM COMBINED PIPELINE ERROR] Fallback due to: {e}")
            clean_raw = [w.strip() for w in raw_query.lower().split() if w.strip()]
            return {
                "is_valid": True,
                "reject_reason": "",
                "combined": clean_raw,
                "primary": clean_raw,
                "secondary": [],
                "expanded": [],
                "constraints": {
                    "exclude_departments": [],
                    "exclude_courses": [],
                    "minor": [],
                    "easy_grading": False
                },
                "minor_query_type": "simple"
            }