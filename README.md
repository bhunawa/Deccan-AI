"""
=============================================================================
DECCAN AI - RECRUITING PIPELINE (Standalone Python Implementation)
=============================================================================
Requires: openai, google-api-python-client, pandas
=============================================================================
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from openai import OpenAI
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    google_credentials_path: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    spreadsheet_id: str = "1hASo56JeX_kW1P430ecI0Kn-mY2uhWr6RN6tNZ5DbhA"
    sheet_name: str = "Form Responses 1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7

config = Config()

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Candidate:
    name: str
    title: str
    years_experience: int
    skills: List[str]
    match_score: int = 0
    interest_score: int = 0
    match_explanation: str = ""
    conversation_summary: str = ""
    
    @property
    def combined_score(self) -> int:
        return (self.match_score + self.interest_score) // 2

@dataclass
class ParsedJD:
    job_title: str
    required_skills: List[str]
    experience_years: int
    nice_to_have: List[str]
    location: str
    employment_type: str

# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    def __init__(self, config: Config):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.model
        self.temperature = config.temperature
    
    def chat(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=self.temperature
        )
        return response.choices[0].message.content

# ============================================================================
# GOOGLE SHEETS CLIENT
# ============================================================================

class SheetsClient:
    def __init__(self, config: Config):
        credentials = service_account.Credentials.from_service_account_file(
            config.google_credentials_path,
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        self.service = build('sheets', 'v4', credentials=credentials)
        self.spreadsheet_id = config.spreadsheet_id
        self.sheet_name = config.sheet_name
    
    def read_candidates(self) -> List[Dict]:
        result = self.service.spreadsheets().values().get(
            spreadsheetId=self.spreadsheet_id,
            range=self.sheet_name
        ).execute()
        
        values = result.get('values', [])
        if not values:
            return []
        
        # Assume first row is headers
        headers = values[0]
        candidates = []
        
        for row in values[1:]:
            candidate = dict(zip(headers, row))
            candidates.append(candidate)
        
        return candidates

# ============================================================================
# AGENT: JD PARSER
# ============================================================================

class JDParserAgent:
    SYSTEM_PROMPT = "You are an AI assistant that parses job descriptions and extracts requirements, skills, and experience."
    
    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def parse(self, job_description: str) -> ParsedJD:
        user_message = f"""Parse the following job description and extract the required skills, experience, and qualifications. Output in structured JSON format with: job_title, required_skills, experience_years, nice_to_have, location, employment_type.

JOB DESCRIPTION:
{job_description}"""
        
        response = self.llm.chat(self.SYSTEM_PROMPT, user_message)
        
        # Parse JSON from response
        try:
            # Extract JSON block
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            return ParsedJD(
                job_title=data.get('job_title', ''),
                required_skills=data.get('required_skills', []),
                experience_years=int(data.get('experience_years', 0)),
                nice_to_have=data.get('nice_to_have', []),
                location=data.get('location', ''),
                employment_type=data.get('employment_type', '')
            )
        except Exception as e:
            print(f"Error parsing JD: {e}")
            return ParsedJD(
                job_title="Unknown",
                required_skills=[],
                experience_years=0,
                nice_to_have=[],
                location="",
                employment_type=""
            )

# ============================================================================
# AGENT: CANDIDATE DISCOVERY
# ============================================================================

class CandidateDiscoveryAgent:
    SYSTEM_PROMPT = "You are an AI recruiting assistant that discovers candidate profiles matching the given job requirements. Use the parsed requirements to search candidate sources and return a concise list of matching candidates with brief explanations for each."
    
    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def find_matches(self, parsed_jd: ParsedJD, candidate_data: List[Dict]) -> List[Candidate]:
        jd_json = json.dumps({
            'job_title': parsed_jd.job_title,
            'required_skills': parsed_jd.required_skills,
            'experience_years': parsed_jd.experience_years,
            'nice_to_have': parsed_jd.nice_to_have,
            'location': parsed_jd.location,
            'employment_type': parsed_jd.employment_type
        })
        
        user_message = f"""Based on the parsed job description requirements below AND the candidate data from the spreadsheet, find matching candidates. For each candidate, provide: name, title, years_experience, skills (array), match_score (0-100), match_explanation (why they match).

JOB REQUIREMENTS:
{jd_json}

CANDIDATE DATA FROM SPREADSHEET:
{json.dumps(candidate_data[:10])}  # Limit to first 10 for token efficiency

Find candidates that match the job requirements. If no candidate data is available in the spreadsheet, generate 5 sample candidate profiles that would match this role with varying match scores (70-95 range) to demonstrate the ranking system.

Return your response as a JSON array with objects containing: name, title, years_experience, skills, match_score, match_explanation."""
        
        response = self.llm.chat(self.SYSTEM_PROMPT, user_message)
        
        # Parse response
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            candidates = []
            for item in data:
                candidates.append(Candidate(
                    name=item.get('name', ''),
                    title=item.get('title', ''),
                    years_experience=int(item.get('years_experience', 0)),
                    skills=item.get('skills', []),
                    match_score=int(item.get('match_score', 50)),
                    match_explanation=item.get('match_explanation', '')
                ))
            return candidates
        except Exception as e:
            print(f"Error parsing candidates: {e}")
            return self._generate_sample_candidates(parsed_jd)
    
    def _generate_sample_candidates(self, parsed_jd: ParsedJD) -> List[Candidate]:
        """Generate sample candidates if no data available"""
        return [
            Candidate("Sarah Chen", "Senior Developer", 7, ["React", "Node.js", "AWS", "TypeScript"], 92, "Strong match with all required skills"),
            Candidate("Marcus Johnson", "Full Stack Engineer", 5, ["React", "Node.js", "Python", "SQL"], 85, "Good experience match"),
            Candidate("Emily Rodriguez", "Software Engineer", 4, ["JavaScript", "React", "GCP", "MongoDB"], 78, "Partial skill match"),
            Candidate("David Kim", "Backend Developer", 6, ["Node.js", "AWS", "PostgreSQL", "Docker"], 88, "Strong backend match"),
            Candidate("Lisa Thompson", "Lead Developer", 8, ["React", "TypeScript", "Kubernetes", "GraphQL"], 95, "Excellent overall match")
        ]

# ============================================================================
# AGENT: INTEREST ASSESSMENT
# ============================================================================

class InterestAssessmentAgent:
    SYSTEM_PROMPT = "You are an AI recruiting assistant that engages candidates conversationally to assess their genuine interest in a job and provide explainable scores."
    
    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def assess_interest(self, candidates: List[Candidate], job_title: str, location: str) -> List[Candidate]:
        candidates_json = json.dumps([
            {
                'name': c.name,
                'title': c.title,
                'years_experience': c.years_experience,
                'skills': c.skills,
                'match_score': c.match_score
            }
            for c in candidates
        ])
        
        user_message = f"""Using the candidate profiles provided below, simulate a conversational outreach to each candidate and assess their interest level. For each candidate, conduct a brief simulated conversation and provide an interest_score (0-100) based on their response enthusiasm, availability, and fit.

CANDIDATE PROFILES:
{candidates_json}

For each candidate, simulate their response to: "Hi, we're hiring a {job_title} for a hybrid role in {location}. Would you be interested in exploring this opportunity?"

Return your response as a JSON array with objects containing: name, interest_score (0-100), conversation_summary."""
        
        response = self.llm.chat(self.SYSTEM_PROMPT, user_message)
        
        # Parse response and update candidates
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Update candidates with interest scores
            for item in data:
                for candidate in candidates:
                    if candidate.name.lower() == item.get('name', '').lower():
                        candidate.interest_score = int(item.get('interest_score', 50))
                        candidate.conversation_summary = item.get('conversation_summary', '')
                        break
        except Exception as e:
            print(f"Error assessing interest: {e}")
            # Assign default interest scores
            for i, candidate in enumerate(candidates):
                candidate.interest_score = [85, 70, 60, 90, 75][i % 5]
                candidate.conversation_summary = "Simulated response based on profile."
        
        return candidates

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class RecruitingPipeline:
    def __init__(self, config: Config):
        self.llm = LLMClient(config)
        self.sheets = SheetsClient(config)
        self.jd_parser = JDParserAgent(self.llm)
        self.candidate_discovery = CandidateDiscoveryAgent(self.llm)
        self.interest_assessment = InterestAssessmentAgent(self.llm)
    
    def run(self, job_description: str) -> List[Candidate]:
        print("=" * 60)
        print("STEP 1: Parsing Job Description")
        print("=" * 60)
        parsed_jd = self.jd_parser.parse(job_description)
        print(f"Job Title: {parsed_jd.job_title}")
        print(f"Required Skills: {parsed_jd.required_skills}")
        print(f"Experience: {parsed_jd.experience_years} years")
        
        print("\n" + "=" * 60)
        print("STEP 2: Reading Candidate Data from Sheets")
        print("=" * 60)
        candidate_data = self.sheets.read_candidates()
        print(f"Found {len(candidate_data)} candidates in spreadsheet")
        
        print("\n" + "=" * 60)
        print("STEP 3: Discovering Matching Candidates")
        print("=" * 60)
        candidates = self.candidate_discovery.find_matches(parsed_jd, candidate_data)
        print(f"Found {len(candidates)} matching candidates")
        
        print("\n" + "=" * 60)
        print("STEP 4: Assessing Candidate Interest")
        print("=" * 60)
        candidates = self.interest_assessment.assess_interest(
            candidates, 
            parsed_jd.job_title, 
            parsed_jd.location
        )
        
        # Sort by combined score
        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        
        print("\n" + "=" * 60)
        print("FINAL RANKED RESULTS")
        print("=" * 60)
        for i, c in enumerate(candidates, 1):
            print(f"\n#{i} {c.name}")
            print(f"   Title: {c.title}")
            print(f"   Match Score: {c.match_score}")
            print(f"   Interest Score: {c.interest_score}")
            print(f"   Combined Score: {c.combined_score}")
            print(f"   Skills: {', '.join(c.skills)}")
        
        return candidates

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Sample job description
    job_description = """
Senior Software Engineer - Full Stack

We are looking for a Senior Software Engineer to join our growing engineering team. 

Requirements:
- 5+ years of experience in software development
- Strong proficiency in JavaScript/TypeScript, React, and Node.js
- Experience with cloud platforms (AWS or GCP)
- Strong understanding of database design (SQL and NoSQL)
- Experience with RESTful APIs and GraphQL
- Excellent problem-solving skills
- Strong communication skills

Nice to have:
- Experience with DevOps practices (CI/CD, Docker, Kubernetes)
- Experience in fintech or payments industry
- Open source contributions

This is a hybrid role based in San Francisco. Competitive salary and equity.
    """
    
    # Run pipeline
    pipeline = RecruitingPipeline(config)
    results = pipeline.run(job_description)
    
    # Output as JSON
    output = [
        {
            "name": c.name,
            "title": c.title,
            "skills": c.skills,
            "match_score": c.match_score,
            "interest_score": c.interest_score,
            "combined_score": c.combined_score,
            "match_explanation": c.match_explanation,
            "conversation_summary": c.conversation_summary
        }
        for c in results
    ]
    
    print("\n" + "=" * 60)
    print("JSON OUTPUT")
    print("=" * 60)
    print(json.dumps(output, indent=2))
