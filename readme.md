# Content Extractor API with Job Title Detection

A FastAPI application that extracts URLs, email addresses, hashtags, and job titles from text content. This API is designed to be deployed on Vercel as a serverless function.

## Features

- Extract URLs with handling for special formatting and markdown
- Extract email addresses
- Extract hashtags
- Detect job titles using NLP techniques
- Lightweight design suitable for serverless deployment

## Requirements

- Python 3.9+
- FastAPI
- SpaCy with the small English model (en_core_web_sm)
- Pydantic for data validation
- Uvicorn for local server

## Local Development

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the API locally:
   ```
   uvicorn main:app --reload
   ```
4. The API will be available at http://localhost:8000
5. Access the auto-generated API documentation at http://localhost:8000/docs

## API Usage

### POST /extract/

Send a POST request with JSON body:

```json
{
  "content": "Your text content to analyze"
}
```

Example response:

```json
{
  "content": "Your text content to analyze",
  "extracted_urls": ["https://example.com"],
  "extracted_emails": ["info@example.com"],
  "extracted_hashtags": ["#JobPosting"],
  "job_title": "Software Engineer"
}
```

## Testing

Run the test script to see API results for sample inputs:

```
python test_api.py
```

## Deployment to Vercel

1. Make sure you have the Vercel CLI installed:
   ```
   npm install -g vercel
   ```

2. Login to Vercel:
   ```
   vercel login
   ```

3. Deploy the project:
   ```
   vercel
   ```

4. For production deployment:
   ```
   vercel --prod
   ```

## How the Job Title Detection Works

The job title detection uses a combination of techniques:

1. Pattern matching with common job title keywords
2. NLP analysis using SpaCy to identify potential job titles based on part-of-speech and named entity recognition
3. Analysis of capitalized terms that might be job titles
4. Examination of hashtags that contain job-related keywords

The model is intentionally kept small to work efficiently in serverless environments.

## Limitations

- The SpaCy model used (en_core_web_sm) is small enough for serverless deployment but may have limited accuracy compared to larger models
- Job title detection is based on patterns and common job keywords, so unusual or specialized job titles might not be detected
- The API focuses on English language content