import openai
import os
import json
import re
from typing import List, Dict, Optional, Any

# Initialize OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    USE_NEW_CLIENT = True
except ImportError:
    # Fallback to legacy client
    openai.api_key = os.getenv("OPENAI_API_KEY")
    USE_NEW_CLIENT = False

def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON substring from text with multiple fallback strategies."""
    if not text:
        return None
    
    try:
        # Strategy 1: Match ```json ... ``` block
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if json_match:
            return json_match.group(1).strip()
        
        # Strategy 2: Match ``` ... ``` block (without json label)
        json_match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Strategy 3: Find first complete JSON object
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    return text[start_idx:i+1].strip()
        
        # Strategy 4: Try to parse the entire text as JSON
        json.loads(text.strip())
        return text.strip()
        
    except Exception as e:
        print(f"JSON extraction error: {e}")
    
    return None

def validate_analysis_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize the analysis output to ensure required fields exist."""
    required_fields = {
        "Summary": "Analysis not available",
        "Verdict": "N/A", 
        "Score": 0,
        "Relevant Skills": []
    }
    
    # Ensure all required fields exist
    for field, default in required_fields.items():
        if field not in data:
            data[field] = default
    
    # Validate verdict
    valid_verdicts = ["Excellent Fit", "Good Fit", "Average Fit", "Weak Fit", "No Fit", "N/A"]
    if data["Verdict"] not in valid_verdicts:
        data["Verdict"] = "N/A"
    
    # Ensure Score is an integer between 0-10
    try:
        score = int(data["Score"])
        data["Score"] = max(0, min(10, score))
    except (ValueError, TypeError):
        data["Score"] = 0
    
    # Ensure Relevant Skills is a list
    if not isinstance(data["Relevant Skills"], list):
        data["Relevant Skills"] = []
    
    # Ensure Summary is a string
    if not isinstance(data["Summary"], str):
        data["Summary"] = str(data["Summary"]) if data["Summary"] else "Analysis not available"
    
    return data

def validate_assessment_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize the assessment output to ensure required fields exist."""
    required_fields = {
        "Overall Assessment": "Assessment not available",
        "Strengths": [],
        "Weaknesses": [],
        "Areas for Improvement": [],
        "Missing Skills": [],
        "Recommendations": [],
        "Overall Score": 0
    }
    
    # Ensure all required fields exist
    for field, default in required_fields.items():
        if field not in data:
            data[field] = default
    
    # Validate Overall Score
    try:
        score = int(data["Overall Score"])
        data["Overall Score"] = max(0, min(10, score))
    except (ValueError, TypeError):
        data["Overall Score"] = 0
    
    # Ensure all list fields are actually lists
    list_fields = ["Strengths", "Weaknesses", "Areas for Improvement", "Missing Skills", "Recommendations"]
    for field in list_fields:
        if not isinstance(data[field], list):
            data[field] = []
    
    return data

def create_analysis_prompt(jd_text: str, context_chunks: List[Dict[str, str]]) -> str:
    """Create a well-structured prompt for job-resume analysis."""
    context = "\n\n".join(f"- {chunk['preview']}" for chunk in context_chunks)
    
    return f"""You are an expert recruiter analyzing how well a candidate's resume matches a job description.

Job Description:
\"\"\"
{jd_text}
\"\"\"

Relevant Resume Snippets:
\"\"\"
{context}
\"\"\"

Analyze the fit between this resume and job description. Consider:
- Technical skills alignment
- Experience relevance  
- Education/certifications match
- Industry experience
- Role responsibilities overlap

Respond ONLY with valid JSON in this exact format:

```json
{{
  "Summary": "A concise 2-3 sentence summary of the candidate's fit for this role",
  "Verdict": "One of: Excellent Fit, Good Fit, Average Fit, Weak Fit, No Fit",
  "Score": 7,
  "Relevant Skills": ["skill1", "skill2", "skill3"]
}}
```

Important: 
- Score must be integer 0-10
- Verdict must be exactly one of the listed options
- Relevant Skills should list 3-8 most important matching skills
- Summary should be professional and specific"""

def create_assessment_prompt(jd_text: str, context_chunks: List[Dict[str, str]]) -> str:
    """Create a prompt for comprehensive resume assessment and improvement recommendations."""
    context = "\n\n".join(f"- {chunk['preview']}" for chunk in context_chunks)
    
    return f"""You are an expert career counselor and recruiter providing comprehensive feedback on a resume for a specific job opportunity.

Job Description:
\"\"\"
{jd_text}
\"\"\"

Resume Content:
\"\"\"
{context}
\"\"\"

Provide a detailed assessment of this resume for the given job, focusing on:
- Current strengths and what the candidate does well
- Areas of weakness that hurt their candidacy
- Specific skills or experiences missing from the job requirements
- Actionable recommendations for improvement
- Overall competitiveness assessment

Respond ONLY with valid JSON in this exact format:

```json
{{
  "Overall Assessment": "2-3 sentences summarizing the candidate's overall fit and competitiveness",
  "Strengths": ["strength1", "strength2", "strength3"],
  "Weaknesses": ["weakness1", "weakness2", "weakness3"],
  "Areas for Improvement": ["improvement1", "improvement2"],
  "Missing Skills": ["skill1", "skill2", "skill3"],
  "Recommendations": ["actionable recommendation1", "actionable recommendation2"],
  "Overall Score": 7
}}
```

Important:
- Overall Score must be integer 0-10
- Include 3-6 items in each list
- Recommendations should be specific and actionable
- Focus on job-relevant improvements"""

def make_openai_request(messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
    """Make OpenAI API request with proper error handling."""
    try:
        if USE_NEW_CLIENT:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"⚠️ OpenAI API error: {e}")
        raise

def analyze_fit(jd_text: str, context_chunks: List[Dict[str, str]], chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Analyze how well a candidate's resume matches a job description.
    
    Args:
        jd_text: Job description text
        context_chunks: List of resume chunks with 'preview' key
        chat_history: Optional chat history for context
    
    Returns:
        Dictionary with analysis results
    """
    if not jd_text or not context_chunks:
        return {
            "Summary": "Insufficient information provided for analysis.",
            "Verdict": "N/A",
            "Score": 0,
            "Relevant Skills": []
        }
    
    # Build messages
    messages = [{"role": "system", "content": "You are an expert recruiter assistant specializing in resume-job fit analysis."}]
    
    if chat_history:
        messages.extend(chat_history)
    
    # Add the analysis prompt
    prompt = create_analysis_prompt(jd_text, context_chunks)
    messages.append({"role": "user", "content": prompt})
    
    try:
        output = make_openai_request(messages, max_tokens=1000)
        print("LLM raw output:\n", output)  # Debug logging
        
        # Extract and parse JSON
        json_text = extract_json_from_text(output)
        if not json_text:
            raise ValueError("Could not extract valid JSON from model output")
        
        parsed_output = json.loads(json_text)
        
        # Validate and return results
        return validate_analysis_output(parsed_output)
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing error: {e}")
        print(f"Attempted to parse: {json_text if 'json_text' in locals() else 'No JSON extracted'}")
        
    except Exception as e:
        print(f"⚠️ API or processing error: {e}")
        if 'output' in locals():
            print(f"Raw output was: {output}")
    
    # Return fallback response
    return {
        "Summary": "Unable to complete analysis due to processing error.",
        "Verdict": "N/A", 
        "Score": 0,
        "Relevant Skills": []
    }

def assess_resume(jd_text: str, context_chunks: List[Dict[str, str]], chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Provide comprehensive assessment of a resume including weaknesses and improvement recommendations.
    
    Args:
        jd_text: Job description text
        context_chunks: List of resume chunks with 'preview' key
        chat_history: Optional chat history for context
    
    Returns:
        Dictionary with assessment results
    """
    if not jd_text or not context_chunks:
        return {
            "Overall Assessment": "Insufficient information provided for assessment.",
            "Strengths": [],
            "Weaknesses": [],
            "Areas for Improvement": [],
            "Missing Skills": [],
            "Recommendations": [],
            "Overall Score": 0
        }
    
    # Build messages
    messages = [{"role": "system", "content": "You are an expert career counselor and recruiter providing comprehensive resume feedback."}]
    
    if chat_history:
        messages.extend(chat_history)
    
    # Add the assessment prompt
    prompt = create_assessment_prompt(jd_text, context_chunks)
    messages.append({"role": "user", "content": prompt})
    
    try:
        output = make_openai_request(messages, max_tokens=1500)
        print("Resume assessment raw output:\n", output)  # Debug logging
        
        # Extract and parse JSON
        json_text = extract_json_from_text(output)
        if not json_text:
            raise ValueError("Could not extract valid JSON from model output")
        
        parsed_output = json.loads(json_text)
        
        # Validate and return results
        return validate_assessment_output(parsed_output)
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing error in assessment: {e}")
        print(f"Attempted to parse: {json_text if 'json_text' in locals() else 'No JSON extracted'}")
        
    except Exception as e:
        print(f"⚠️ API or processing error in assessment: {e}")
        if 'output' in locals():
            print(f"Raw output was: {output}")
    
    # Return fallback response
    return {
        "Overall Assessment": "Unable to complete assessment due to processing error.",
        "Strengths": [],
        "Weaknesses": [],
        "Areas for Improvement": [],
        "Missing Skills": [],
        "Recommendations": [],
        "Overall Score": 0
    }