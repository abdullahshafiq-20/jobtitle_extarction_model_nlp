# main.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, EmailStr
import re
import spacy
from typing import List, Optional, Dict, Any
import logging
import os
import asyncio
from datetime import datetime
from functools import lru_cache
from email_valid import EmailValidator

email_validator = EmailValidator()



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load spaCy model with caching for better performance
@lru_cache(maxsize=1)
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("Model not found. Install with: python -m spacy download en_core_web_sm")
        return spacy.blank("en")  # Create a blank model as fallback

# Initialize FastAPI with metadata
app = FastAPI(
    title="Content Extractor API",
    description="API for extracting URLs, emails, hashtags, and job titles from text content",
    version="1.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContentRequest(BaseModel):
    content: str = Field(..., description="The text content to analyze")
    
    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

class ContentResponse(BaseModel):
    content: str
    extracted_urls: List[str]
    extracted_emails: List[str]
    extracted_hashtags: List[str]
    job_title: Optional[str] = None
    stats: Dict[str, Any]

# Email validation models
class EmailValidationRequest(BaseModel):
    email: EmailStr
    check_smtp: bool = False
    timeout: int = 10

class ValidationResponse(BaseModel):
    email: str
    is_valid: bool
    domain_exists: bool
    mx_records: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []
    smtp_valid: Optional[bool] = None
    validation_time: float
    timestamp: str

class BulkEmailRequest(BaseModel):
    emails: List[EmailStr]
    check_smtp: bool = False
    timeout: int = 10

class BulkValidationResponse(BaseModel):
    total: int
    valid: int
    invalid: int
    processing_time: float
    results: List[ValidationResponse]

# Dependency for NLP model
def get_nlp():
    return load_nlp_model()

@app.post("/extract/", response_model=ContentResponse, status_code=status.HTTP_200_OK)
async def extract_content_info(request: ContentRequest, nlp=Depends(get_nlp)):
    content = request.content
    
    try:
        # Extract URLs - Improved regex for better accuracy
        url_regex = r'(?:https?:\/\/|www\.)(?:[-\w.]+)(?:\/[-\w.\/?\%&=+]*)?|(?:[-\w]+\.(?:com|org|edu|gov|net|io|dev|co|us|uk|de|ru|jp|fr|in|au|ca|br|es|it|nl|se|no|fi|dk|ch|be|at|pt|pl|cz|hu|gr|ro|za|nz|mx|ar|cl|co|pe|ve|tr|sa|ae|eg|za|ph|my|sg|th|vn|id))(?:\/[-\w.\/?\%&=+]*)?'
        extracted_urls = re.findall(url_regex, content, re.IGNORECASE)
        
        # Clean URLs
        cleaned_urls = []
        for url in extracted_urls:
            # Remove trailing punctuation
            url = re.sub(r'[.,;:!?)]*$', '', url)
            # Remove markdown formatting
            url = url.replace('**', '').replace('__', '')
            # Add https:// prefix to www. URLs if missing
            if url.startswith('www.') and not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            cleaned_urls.append(url)
        
        # Extract emails - Improved regex for international domains
        email_regex = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
        extracted_emails = re.findall(email_regex, content)
        
        # Validate emails to reduce false positives
        validated_emails = []
        for email in extracted_emails:
            # Simple validation to avoid false positives
            if re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email):
                validated_emails.append(email)
        
        # Extract hashtags with improved handling
        hashtag_regex = r'#[\w\d]+'
        extracted_hashtags = re.findall(hashtag_regex, content)
        
        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(cleaned_urls))
        unique_emails = list(dict.fromkeys(validated_emails))
        unique_hashtags = list(dict.fromkeys(extracted_hashtags))
        
        # Extract job title
        job_title = extract_job_title(content, nlp)
        
        # Compile statistics
        stats = {
            "total_urls": len(unique_urls),
            "total_emails": len(unique_emails),
            "total_hashtags": len(unique_hashtags),
            "content_length": len(content),
            "word_count": len(content.split()),
        }
        
        return {
            "content": content,
            "extracted_urls": unique_urls,
            "extracted_emails": unique_emails,
            "extracted_hashtags": unique_hashtags,
            "job_title": job_title,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error processing content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing content: {str(e)}"
        )

def extract_job_title(content, nlp):
    """
    Extract potential job titles from the content using spaCy NER,
    pattern matching, and contextual analysis.
    """
    # Enhanced list of job title keywords
    job_title_keywords = [
        # Engineering & Development
        "engineer", "developer", "architect", "programmer", "coder", "devops", 
        "sre", "qa", "tester", "automation", "frontend", "backend", "fullstack", 
        "mobile", "embedded", "firmware", "hardware", "software", "systems", 
        "network", "infrastructure", "platform", "cloud", "security", "database", 
        "data", "machine", "learning", "ai", "ml", "nlp", "computer", "robotics", 
        "iot", "blockchain", "web3", "crypto", "game", "unity", "unreal", 
        "cybersecurity", "infosec", "pentester", "ethical", "hacker", "devsecops",
        "reliability", "scrum", "agile", "kanban", "django", "node", "react", "vue", 
        "angular", "typescript", "javascript", "java", "python", "ruby", "golang", "rust",
        "kubernetes", "docker", "microservices", "serverless", "lambda", "aws", "azure", "gcp",
        "terraform", "ci/cd", "jenkins", "ansible", "puppet", "chef", "kubernetes", "helm",

        # Management & Leadership
        "manager", "director", "head", "chief", "lead", "supervisor", "coordinator",
        "administrator", "executive", "officer", "president", "vice", "vp", "ceo",
        "cto", "cfo", "coo", "cio", "cmo", "cpo", "cso", "founder", "co-founder",
        "owner", "partner", "principal", "chairman", "board", "trustee", "governor",
        "evangelist", "ambassador", "advocate", "steward", "chancellor", "provost", 
        "dean", "chairperson", "executive", "leadership", "management", "mentor",
        
        # Business & Operations
        "analyst", "consultant", "specialist", "expert", "professional", 
        "coordinator", "assistant", "associate", "representative", "agent", 
        "strategist", "planner", "researcher", "operations", "logistics", 
        "supply", "chain", "procurement", "sourcing", "quality", "compliance", 
        "risk", "audit", "governance", "scrum", "master", "agile", "coach",
        "delivery", "implementation", "solutions", "technical", "presales", "postsales",
        "revenue", "optimization", "onboarding", "customer", "success", "experience",
        "performance", "efficiency", "productivity", "process", "improvement",
        
        # Creative & Design
        "designer", "artist", "creator", "writer", "editor", "producer", "director",
        "photographer", "videographer", "animator", "illustrator", "architect",
        "graphic", "ui", "ux", "interaction", "motion", "3d", "visual", "brand",
        "content", "copy", "creative", "art", "digital", "web", "print", "publishing",
        "industrial", "product", "fashion", "interior", "exhibition", "game", "level",
        "character", "environment", "narrative", "storyboard", "typographer", "layout",
        
        # Sales & Marketing
        "sales", "marketing", "brand", "product", "project", "business", "account",
        "customer", "client", "partner", "relationship", "growth", "acquisition",
        "retention", "development", "strategy", "campaign", "social", "media",
        "content", "seo", "sem", "ppc", "analytics", "research", "market", "affiliate",
        "influencer", "pr", "public", "relations", "communications", "spokesperson",
        "outreach", "email", "digital", "offline", "advertising", "promotional", "merchandising",
        "trade", "branding", "positioning", "evangelist", "advocate", "loyalty", "omnichannel",
        
        # Data & Analytics
        "data", "analytics", "scientist", "researcher", "statistician", "bi",
        "intelligence", "mining", "machine", "learning", "ai", "ml", "nlp",
        "big", "warehouse", "lake", "pipeline", "etl", "visualization", "reporting",
        "forecasting", "prediction", "modeling", "optimization", "research",
        "econometrician", "actuary", "quant", "quantitative", "computational", "bioinformatics", 
        "genomics", "informatics", "operations", "algorithms", "deep", "reinforcement", 
        "computer", "vision", "natural", "language", "processing", "speech", "recognition",
        
        # Support & Service
        "support", "service", "help", "customer", "client", "success", "operations",
        "facilities", "maintenance", "security", "compliance", "legal", "hr",
        "recruiter", "talent", "acquisition", "benefits", "compensation", "training",
        "development", "learning", "education", "facilities", "maintenance",
        "helpdesk", "technician", "concierge", "ambassador", "representative", "agent",
        "operator", "call", "center", "remote", "field", "desktop", "onsite", "virtual",
        "tier", "escalation", "incident", "problem", "change", "ticket", "request",
        
        # Education & Training
        "teacher", "professor", "instructor", "trainer", "educator", "mentor",
        "coach", "tutor", "facilitator", "curator", "librarian", "academic",
        "researcher", "lecturer", "adjunct", "dean", "principal", "headmaster",
        "curriculum", "syllabus", "course", "lesson", "workshop", "seminar", "webinar",
        "teaching", "assistant", "graduate", "undergraduate", "postdoctoral", "scholar",
        "fellow", "doctorate", "phd", "masters", "bachelors", "tenure", "track",
        
        # Healthcare & Medical
        "doctor", "nurse", "physician", "surgeon", "therapist", "counselor",
        "psychologist", "psychiatrist", "dentist", "pharmacist", "technician",
        "practitioner", "specialist", "resident", "intern", "attending",
        "clinician", "clinical", "medical", "health", "healthcare", "occupational",
        "physical", "speech", "radiation", "respiratory", "surgical", "anesthesiologist",
        "cardiologist", "dermatologist", "endocrinologist", "gastroenterologist", "neurologist",
        "oncologist", "ophthalmologist", "pediatrician", "radiologist", "urologist",
        
        # Finance & Accounting
        "accountant", "auditor", "banker", "broker", "trader", "analyst",
        "controller", "treasurer", "actuary", "underwriter", "claims", "tax",
        "investment", "portfolio", "fund", "risk", "compliance", "treasury",
        "financial", "fiscal", "monetary", "budgeting", "forecast", "revenue",
        "cost", "expense", "profit", "loss", "balance", "sheet", "income", "statement",
        "cash", "flow", "equity", "debt", "capital", "asset", "liability", "dividend",
        "merger", "acquisition", "valuation", "underwriting", "private", "equity", "venture",
        
        # Legal & Compliance
        "lawyer", "attorney", "counsel", "paralegal", "legal", "compliance", 
        "regulatory", "patent", "intellectual", "property", "copyright", "trademark", 
        "litigation", "corporate", "contract", "privacy", "gdpr", "hipaa", "sox", 
        "appellate", "judiciary", "arbitrator", "mediator", "notary", "legislator",
        
        # Seniority Levels
        "senior", "junior", "principal", "staff", "associate", "entry", "mid",
        "lead", "chief", "head", "vice", "assistant", "deputy", "executive",
        "fellow", "distinguished", "emeritus", "honorary", "visiting",
        "trainee", "apprentice", "intern", "graduate", "entry-level", "c-level",
        "director", "manager", "supervisor", "individual", "contributor", "specialist",
    ]

    # Process the content with spaCy
    doc = nlp(content)

    # Method 1: Look for context clues like "job title:", "position:", etc.
    context_patterns = [
        r'(?:job\s+title|position|role|title)(?:\s+is|\s*:\s*)([^\.!?\n]+)',
        r'(?:hiring|looking\s+for|recruiting|seeking|searching\s+for|need)(?:\s+a|\s+an|\s+)([^\.!?\n]+)',
        r'(?:apply|application|candidate|candidates)(?:\s+for\s+the\s+)([^\.!?\n]+)',
        r'(?:job|position|career|vacancy|opening)\s+(?:opening|opportunity|posting|listing|advertisement|advert|description)\s+for\s+([^\.!?\n]+)',
        r'(?:we\s+are\s+hiring\s+a|we\s+need\s+a|join\s+us\s+as\s+a|become\s+our|join\s+our\s+team\s+as\s+a)([^\.!?\n]+)',
        r'(?:career|job)\s+(?:opportunity|opening)(?:\s+as\s+a|\s+as\s+)([^\.!?\n]+)',
        r'(?:work\s+with\s+us\s+as\s+a|work\s+as\s+a|employed\s+as\s+a|employed\s+as\s+)([^\.!?\n]+)',
        r'(?:current\s+role|current\s+position|current\s+job)(?:\s+is|\s*:\s*)([^\.!?\n]+)',
        r'(?:I\s+am\s+a|I\s+work\s+as\s+a|I\s+currently\s+serve\s+as\s+a)([^\.!?\n]+)',
        r'(?:experienced|skilled|professional|certified|qualified)(?:\s+in|\s+as\s+a|\s+as\s+an|\s+)([^\.!?\n]+)'
    ]

    for pattern in context_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            # Take the first match and clean it
            title_match = matches[0].strip()
            
            # If the match is too long, it might be a false positive
            if len(title_match.split()) <= 8:  # Increased from 6 to 8 to capture more complex titles
                # Check if match contains any job keywords
                if any(keyword in title_match.lower() for keyword in job_title_keywords):
                    return title_match.strip()
    
    # Method 2: Look for named entities that might be job titles
    for ent in doc.ents:
        if ent.label_ == "ORG" or ent.label_ == "PRODUCT":
            text = ent.text.lower()
            # Check if entity contains job keywords
            if any(keyword in text for keyword in job_title_keywords):
                return ent.text
    
    # Method 3: Use patterns to find common job title formats
    patterns = [
        # QA/Testing patterns
        r'(?:qa|quality|test|testing)\s+(?:engineer|analyst|specialist|lead|manager|director|automation|manual|performance)(?:\s+\([^)]+\))?',
        r'(?:software|automation|manual|performance)\s+(?:qa|quality|test|testing)\s+(?:engineer|analyst|specialist|lead|manager)',
        r'(?:senior|junior|lead|principal|staff|associate|mid|entry-level)\s+(?:qa|quality|test|testing)\s+(?:engineer|analyst|specialist|lead)',
        
        # Engineering patterns
        r'(?:senior|junior|lead|principal|staff|associate|mid|entry-level)\s+[a-zA-Z]+\s+(?:engineer|developer|architect|analyst|designer)',
        r'(?:software|hardware|systems|network|security|cloud|data|ai|ml|cyber|information|application)\s+(?:engineer|developer|architect|analyst|specialist)',
        r'(?:frontend|backend|fullstack|full-stack|mobile|web|devops|ui|ux|devsecops|site\s+reliability|platform|infrastructure)\s+(?:developer|engineer|architect|specialist)',
        r'(?:senior|junior|lead|principal|staff|mid|entry-level)\s+(?:software|hardware|systems|network)\s+(?:engineer|developer|architect)',
        r'(?:c\#|java|python|javascript|typescript|ruby|golang|rust|php|scala|swift|kotlin|c\+\+)\s+(?:engineer|developer|architect|programmer)',
        r'(?:react|angular|vue|node|django|spring|laravel|flask|rails|express)\s+(?:engineer|developer|architect|specialist)',
        r'(?:aws|azure|gcp|cloud)\s+(?:engineer|developer|architect|specialist|consultant)',
        
        # Data & Analytics patterns
        r'(?:data|machine|artificial|intelligence|business|big\s+data|ai)\s+(?:scientist|engineer|analyst|architect|modeler)',
        r'(?:business|systems|data|marketing|financial|sales|operations|hr)\s+(?:analyst|architect|engineer|specialist)',
        r'(?:senior|junior|lead|principal|staff|associate)\s+(?:data|business|systems)\s+(?:analyst|architect|engineer|scientist)',
        r'(?:bi|business\s+intelligence|reporting|tableau|power\s+bi|qlik)\s+(?:developer|engineer|analyst|specialist)',
        r'(?:etl|data\s+pipeline|data\s+warehouse|data\s+lake)\s+(?:developer|engineer|architect|specialist)',
        
        # Management patterns
        r'(?:product|project|program|business|technical|engineering|marketing|sales)\s+(?:manager|director|lead|owner|head)',
        r'(?:senior|junior|lead|principal|associate|vp|vice\s+president\s+of)\s+(?:product|project|program)\s+(?:manager|director|lead)',
        r'(?:chief|vice|associate|assistant|deputy|executive|global|regional)\s+(?:technology|information|data|product|marketing|financial|operations|human\s+resources|customer)\s+(?:officer|director|manager)',
        r'(?:head|director)\s+of\s+(?:engineering|product|marketing|sales|design|operations|finance|hr|legal)',
        
        # Design patterns
        r'(?:ui|ux|user\s+interface|user\s+experience|product|interaction|visual|graphic)\s+(?:designer|developer|architect|specialist|researcher)',
        r'(?:senior|junior|lead|principal|staff|associate)\s+(?:ui|ux|graphic|web|product|industrial|fashion)\s+(?:designer|developer)',
        r'(?:creative|art)\s+(?:director|lead|manager|head)',
        r'(?:brand|visual|design|product)\s+(?:strategist|specialist|consultant|manager)',
        
        # Marketing & Sales patterns
        r'(?:digital|content|brand|product|growth|performance|social\s+media)\s+(?:marketing|specialist|strategist|manager|coordinator)',
        r'(?:seo|sem|ppc|paid\s+search|organic|content|email)\s+(?:specialist|strategist|manager|analyst)',
        r'(?:sales|account|business\s+development|customer\s+success)\s+(?:representative|executive|manager|director)',
        r'(?:senior|junior|lead|principal|associate)\s+(?:sales|marketing|account)\s+(?:manager|executive|representative)',
        
        # Healthcare patterns
        r'(?:medical|clinical|health|healthcare|patient)\s+(?:specialist|coordinator|manager|director|administrator)',
        r'(?:registered|licensed|practical)\s+(?:nurse|therapist|counselor|technician)',
        r'(?:physical|occupational|speech|respiratory)\s+(?:therapist|assistant|technician)',
        
        # Finance patterns
        r'(?:financial|investment|tax|audit|accounting)\s+(?:analyst|specialist|manager|advisor|consultant)',
        r'(?:senior|junior|lead|principal|associate)\s+(?:accountant|auditor|analyst|advisor)',
        r'(?:portfolio|fund|asset|wealth|investment)\s+(?:manager|analyst|advisor|consultant)',
        
        # Legal patterns
        r'(?:corporate|litigation|patent|intellectual\s+property|privacy|compliance)\s+(?:lawyer|attorney|counsel|specialist)',
        r'(?:general|associate|assistant|senior)\s+(?:counsel|attorney)',
        r'(?:legal|compliance|regulatory|governance)\s+(?:specialist|manager|director|officer)',
        
        # Internship patterns
        r'(?:summer|winter|fall|spring|research|graduate|undergraduate)?\s*(?:internship|intern|trainee|apprentice)\s+(?:at|with|for)?\s*(?:[a-zA-Z\s]+)',
        r'(?:graduate|undergraduate|phd|masters|junior)\s+(?:internship|intern|research|trainee)\s+(?:position|opportunity|program|role)',
        
        # Remote work patterns
        r'(?:remote|virtual|work\s+from\s+home|telecommute|distributed)\s+(?:engineer|developer|designer|manager|specialist|analyst)',
        
        # Freelance patterns
        r'(?:freelance|contract|independent|consultant|temporary|interim)\s+(?:engineer|developer|designer|writer|editor|specialist|consultant)',
        
        # Education patterns
        r'(?:adjunct|assistant|associate|full|visiting)\s+(?:professor|lecturer|instructor|faculty)',
        r'(?:teacher|instructor|tutor|coach)\s+of\s+(?:math|science|english|history|computer|programming|art)',
        
        # Academic/Research patterns
        r'(?:research|postdoctoral|graduate|phd|doctoral)\s+(?:fellow|associate|assistant|scientist|scholar)',
        r'(?:lab|laboratory|research)\s+(?:manager|director|coordinator|technician|assistant)',
        
        # Executive patterns
        r'(?:c-suite|c-level|executive|chief)\s+(?:officer|executive|director|advisor)',
        r'(?:founder|co-founder|president|vice\s+president|chairperson)\s+(?:and|&)?\s*(?:ceo|cto|cfo|coo|cmo)?',
        
        # Human Resources patterns
        r'(?:hr|human\s+resources|talent|recruitment|people|organizational)\s+(?:specialist|coordinator|manager|director|partner)',
        r'(?:talent|recruitment|learning|development|compensation|benefits)\s+(?:specialist|coordinator|manager|director)'
    ]
    
    # Try to find job titles using the patterns
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            # Take the longest match (likely most specific)
            matches.sort(key=len, reverse=True)
            return matches[0].strip().title()
    
    # Method 4: Look for lines that might contain job titles with NLP
    for sent in doc.sents:
        sent_text = sent.text.lower()
        
        # Count job keywords in sentence
        keyword_count = sum(1 for keyword in job_title_keywords if keyword in sent_text)
        
        # If sentence has multiple job keywords, it might contain a job title
        if keyword_count >= 2:
            # Find noun chunks that might be job titles
            for chunk in sent.noun_chunks:
                chunk_text = chunk.text.lower()
                if any(keyword in chunk_text for keyword in job_title_keywords):
                    # Extend to include preceding adjectives if available
                    start = chunk.start
                    for i in range(chunk.start-1, -1, -1):
                        if doc[i].pos_ == "ADJ" and i >= 0:
                            start = i
                        else:
                            break
                    
                    potential_title = doc[start:chunk.end].text
                    
                    # Verify it's not too long (likely not a job title if too many words)
                    if len(potential_title.split()) <= 5:
                        return potential_title.title()
    
    # Method 5: Look for hashtags that might indicate job titles
    hashtag_matches = []
    for hashtag in re.findall(r'#[\w\d]+', content):
        hashtag_text = hashtag[1:].lower()
        if any(keyword in hashtag_text for keyword in job_title_keywords):
            hashtag_matches.append(hashtag)
    
    if hashtag_matches:
        # Take the most promising hashtag
        best_match = max(hashtag_matches, key=lambda h: sum(1 for k in job_title_keywords if k in h.lower()))
        
        # Format hashtag as job title
        hashtag_text = best_match[1:]  # Remove #
        
        # Try different word separation techniques
        if re.search(r'[A-Z]', hashtag_text[1:]):  # CamelCase
            formatted_title = ' '.join(re.findall('[A-Z][a-z]*', hashtag_text))
            if formatted_title:
                return formatted_title
        
        if '_' in hashtag_text:  # snake_case
            return ' '.join(word.capitalize() for word in hashtag_text.split('_'))
            
        # Split by number boundaries
        words = re.findall(r'[a-zA-Z]+|\d+', hashtag_text)
        return ' '.join(word.capitalize() for word in words)
    
    # No job title found
    return None

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Welcome to Content Extractor API v1.1.0",
        "description": "Use POST /extract/ endpoint to analyze content",
        "documentation": "/docs",
        "endpoints": {
            "extract": "/extract/ (POST)",
            "health": "/health/ (GET)"
        }
    }

@app.get("/health/", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.1.0"}

@app.post("/validate", response_model=ValidationResponse)
async def validate_email(request: EmailValidationRequest):
    """
    Validate a single email address
    
    - **email**: Email address to validate
    - **check_smtp**: Whether to perform SMTP validation (slower but more thorough)
    - **timeout**: Timeout in seconds for DNS and SMTP checks
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        result = await email_validator.validate_email(
            request.email, 
            request.check_smtp, 
            request.timeout
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        suggestions = email_validator.get_suggestions(request.email)
        
        return ValidationResponse(
            email=result.email,
            is_valid=result.is_valid,
            domain_exists=result.domain_exists,
            mx_records=result.mx_records,
            errors=result.errors,
            warnings=result.warnings,
            suggestions=suggestions,
            smtp_valid=result.smtp_valid,
            validation_time=round(processing_time, 3),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Validation error for {request.email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/validate/bulk", response_model=BulkValidationResponse)
async def validate_bulk_emails(request: BulkEmailRequest):
    """
    Validate multiple email addresses
    
    - **emails**: List of email addresses to validate
    - **check_smtp**: Whether to perform SMTP validation
    - **timeout**: Timeout in seconds for each validation
    """
    if len(request.emails) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 emails allowed per request")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Process emails concurrently
        tasks = [
            email_validator.validate_email(email, request.check_smtp, request.timeout)
            for email in request.emails
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = []
        valid_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle individual failures
                response = ValidationResponse(
                    email=request.emails[i],
                    is_valid=False,
                    domain_exists=False,
                    mx_records=[],
                    errors=[f"Validation failed: {str(result)}"],
                    warnings=[],
                    suggestions=[],
                    validation_time=0,
                    timestamp=datetime.now().isoformat()
                )
            else:
                if result.is_valid:
                    valid_count += 1
                
                suggestions = email_validator.get_suggestions(result.email)
                response = ValidationResponse(
                    email=result.email,
                    is_valid=result.is_valid,
                    domain_exists=result.domain_exists,
                    mx_records=result.mx_records,
                    errors=result.errors,
                    warnings=result.warnings,
                    suggestions=suggestions,
                    smtp_valid=result.smtp_valid,
                    validation_time=0,  # Individual timing not tracked in bulk
                    timestamp=datetime.now().isoformat()
                )
            
            responses.append(response)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return BulkValidationResponse(
            total=len(request.emails),
            valid=valid_count,
            invalid=len(request.emails) - valid_count,
            processing_time=round(processing_time, 3),
            results=responses
        )
    
    except Exception as e:
        logger.error(f"Bulk validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk validation failed: {str(e)}")

@app.get("/validate/{email}")
async def quick_validate(email: str):
    """
    Quick email validation (GET request for simple checks)
    Only performs format and DNS validation, no SMTP
    """
    try:
        result = await email_validator.validate_email(email, check_smtp=False, timeout=5)
        suggestions = email_validator.get_suggestions(email)
        
        return {
            "email": result.email,
            "is_valid": result.is_valid,
            "domain_exists": result.domain_exists,
            "errors": result.errors,
            "suggestions": suggestions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info")
    
    logger.info(f"Starting Content Extractor API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=log_level)