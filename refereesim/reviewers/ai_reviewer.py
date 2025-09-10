"""AI reviewer that analyzes papers for errors"""

import os
import json
import time
import requests
from typing import List, Dict, Any
from datetime import datetime
import cohere
# IMPORTANT: KEEP THIS COMMENT - Using python_gemini blueprint integration
from google import genai
from google.genai import types

from ..models import ReviewResult, ReviewFinding


class AIReviewer:
    """AI-powered paper reviewer that finds errors in research papers"""
    
    def __init__(self, model_name: str = "command-r", cache_dir: str = "./cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Determine API provider based on model name
        if self._is_hyperbolic_model(model_name):
            self.api_provider = "hyperbolic"
            self.hyperbolic_api_key = os.getenv("HYPERBOLIC_API_KEY")
            if not self.hyperbolic_api_key:
                raise ValueError("HYPERBOLIC_API_KEY environment variable not set")
            self.hyperbolic_url = "https://api.hyperbolic.xyz/v1/chat/completions"
        elif self._is_cohere_model(model_name):
            self.api_provider = "cohere"
            cohere_key = os.getenv("COHERE_API_KEY")
            if not cohere_key:
                raise ValueError("COHERE_API_KEY environment variable not set")
            self.cohere_client = cohere.ClientV2(api_key=cohere_key)
        elif self._is_gemini_model(model_name):
            self.api_provider = "gemini"
            gemini_key = os.getenv("GENAI_API_KEY")
            if not gemini_key:
                raise ValueError("GENAI_API_KEY environment variable not set")
            self.gemini_client = genai.Client(api_key=gemini_key)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use Cohere, Hyperbolic, or Gemini models only.")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def _is_hyperbolic_model(self, model_name: str) -> bool:
        """Check if model is from Hyperbolic API"""
        hyperbolic_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "deepseek-ai/DeepSeek-R1-0528", 
            "deepseek-ai/DeepSeek-R1",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "deepseek-ai/DeepSeek-V3"
        ]
        return model_name in hyperbolic_models
    
    def _is_cohere_model(self, model_name: str) -> bool:
        """Check if model is from Cohere API"""
        cohere_models = [
            "command-a-03-2025",
            "command-r"
        ]
        return model_name in cohere_models
    
    def _is_gemini_model(self, model_name: str) -> bool:
        """Check if model is from Gemini API"""
        gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash-preview"
        ]
        return model_name in gemini_models
    
    def review_paper(self, paper_id: str, paper_content: str, 
                    prompt_style: str = "standard") -> ReviewResult:
        """Review a paper and return findings"""
        
        # Check cache first
        cache_key = f"{paper_id}_{self.model_name}_{prompt_style}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                return ReviewResult(**cached_data)
        
        # Generate review prompt
        prompt = self._generate_prompt(paper_content, prompt_style)
        
        # Call appropriate API
        try:
            if self.api_provider == "hyperbolic":
                response_text = self._call_hyperbolic_api(prompt)
            elif self.api_provider == "cohere":
                response_text = self._call_cohere_api(prompt)
            elif self.api_provider == "gemini":
                response_text = self._call_gemini_api(prompt)
            else:
                raise ValueError(f"Unsupported API provider: {self.api_provider}")
            
            # Parse the response
            findings, overall_assessment = self._parse_response(response_text)
            
            # Create review result
            review_result = ReviewResult(
                paper_id=paper_id,
                model_name=self.model_name,
                prompt_style=prompt_style,
                findings=findings,
                overall_assessment=overall_assessment,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(review_result.model_dump(), f, indent=2)
            
            return review_result
            
        except Exception as e:
            print(f"Error calling {self.api_provider} API: {e}")
            # Return empty result on error
            return ReviewResult(
                paper_id=paper_id,
                model_name=self.model_name,
                prompt_style=prompt_style,
                findings=[],
                overall_assessment=f"Error during review: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _call_hyperbolic_api(self, prompt: str) -> str:
        """Call Hyperbolic API with the given prompt"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.hyperbolic_api_key}"
        }
        
        # Configure model-specific parameters
        if "deepseek" in self.model_name.lower():
            temperature = 0.1
            top_p = 0.9
            max_tokens = 508
        elif "llama" in self.model_name.lower():
            temperature = 0.7
            top_p = 0.9
            max_tokens = 512
        else:  # gpt-oss models
            temperature = 0.7
            top_p = 0.8
            max_tokens = 512
        
        data = {
            "messages": [
                {"role": "system", "content": "You are an expert scientific reviewer tasked with finding errors in research papers."},
                {"role": "user", "content": prompt}
            ],
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        response = requests.post(self.hyperbolic_url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _call_cohere_api(self, prompt: str) -> str:
        """Call Cohere API with the given prompt"""
        try:
            # Use simple message format for Cohere v2
            response = self.cohere_client.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Safely extract text from response 
            if hasattr(response, 'message') and response.message:
                if hasattr(response.message, 'content') and response.message.content:
                    content_parts = response.message.content
                    if content_parts and len(content_parts) > 0:
                        # Handle different content types safely
                        first_part = content_parts[0]
                        if hasattr(first_part, 'text') and first_part.text:
                            return first_part.text
            return "No response content"
            
        except Exception as e:
            print(f"Error calling Cohere API: {e}")
            return f"Error calling Cohere API: {str(e)}"
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API using the python_gemini blueprint"""
        try:
            # Simplified Gemini API call based on blueprint
            system_prompt = "You are an expert scientific reviewer tasked with finding errors in research papers."
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            
            return response.text or "No response content"
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error calling Gemini API: {str(e)}"
    
    def _generate_prompt(self, paper_content: str, prompt_style: str) -> str:
        """Generate review prompt based on style"""
        
        base_instructions = """
Please carefully review the following research paper and identify any errors or problems you can find.

Look specifically for:
1. Statistical errors (wrong p-values, incorrect test statistics, etc.)
2. Calculation errors (numbers that don't add up, wrong percentages, etc.) 
3. Data inconsistencies (sample sizes that don't match, contradictory claims)
4. Methodological issues (data leakage, inappropriate tests, etc.)
5. Table/text mismatches (claims that contradict the data shown)
6. Citation problems (non-existent or suspicious references)

For each error you find, provide:
- Category: The type of error from the list above
- Location: Where in the paper (section, table, line)
- Quoted text: The exact problematic text
- Explanation: Why this is wrong and what the correct approach would be
- Confidence: Your confidence level (0.0 to 1.0) that this is actually an error

Format your response as a list of findings, followed by an overall assessment.
"""
        
        if prompt_style == "standard":
            return f"{base_instructions}\n\nPaper to review:\n{paper_content}"
        else:
            return f"{base_instructions}\n\nPaper to review:\n{paper_content}"
    
    def _parse_response(self, response_text: str) -> tuple[List[ReviewFinding], str]:
        """Parse AI response into structured findings"""
        
        findings = []
        overall_assessment = "Review completed."
        
        # Simple parsing - look for error indicators
        lines = response_text.split('\n')
        current_finding = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for error indicators
            if any(keyword in line.lower() for keyword in ['error', 'wrong', 'incorrect', 'mismatch']):
                if current_finding:
                    # Save previous finding
                    finding = ReviewFinding(
                        category=current_finding.get("category", "unknown"),
                        location=current_finding.get("location", "unknown"),
                        quoted_text=current_finding.get("quoted_text", line),
                        explanation=current_finding.get("explanation", line),
                        confidence=0.6
                    )
                    findings.append(finding)
                
                # Start new finding
                current_finding = {
                    "category": "unknown",
                    "location": "unknown", 
                    "quoted_text": line,
                    "explanation": line
                }
            
            # Update current finding based on keywords
            if "category:" in line.lower():
                current_finding["category"] = line.split(":", 1)[1].strip()
            elif "location:" in line.lower():
                current_finding["location"] = line.split(":", 1)[1].strip()
            elif "explanation:" in line.lower():
                current_finding["explanation"] = line.split(":", 1)[1].strip()
            elif "overall" in line.lower() and "assessment" in line.lower():
                overall_assessment = line
        
        # Save last finding
        if current_finding:
            finding = ReviewFinding(
                category=current_finding.get("category", "unknown"),
                location=current_finding.get("location", "unknown"),
                quoted_text=current_finding.get("quoted_text", ""),
                explanation=current_finding.get("explanation", ""),
                confidence=0.6
            )
            findings.append(finding)
        
        return findings, overall_assessment


class MultiModelReviewer:
    """Reviewer that uses multiple AI models"""
    
    def __init__(self, models: List[str] = None, cache_dir: str = "./cache"):
        # Default to working models only (no OpenAI)
        if models is None:
            models = [
                "command-r",
                "command-a-03-2025", 
                "gemini-2.5-flash",
                "openai/gpt-oss-120b",
                "meta-llama/Meta-Llama-3.1-70B-Instruct"
            ]
        
        self.models = models
        self.reviewers = {}
        
        # Initialize reviewers for each model
        for model in models:
            try:
                self.reviewers[model] = AIReviewer(model, cache_dir)
                print(f"✅ Initialized {model}")
            except Exception as e:
                print(f"❌ Failed to initialize {model}: {e}")
    
    def review_paper_all_models(self, paper_id: str, paper_content: str, 
                               prompt_styles: List[str] = None) -> List[ReviewResult]:
        """Review paper with all available models"""
        results = []
        
        for model in self.models:
            if model not in self.reviewers:
                print(f"Skipping {model} - not initialized")
                continue
                
            for style in prompt_styles:
                try:
                    print(f"Reviewing {paper_id} with {model} ({style})...")
                    result = self.reviewers[model].review_paper(paper_id, paper_content, style)
                    results.append(result)
                except Exception as e:
                    print(f"Error reviewing {paper_id} with {model}: {e}")
        
        return results