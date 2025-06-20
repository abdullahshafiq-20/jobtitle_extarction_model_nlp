import re
import dns.resolver
import smtplib
import socket
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class EmailValidationRequest(BaseModel):
    email: str
    check_smtp: bool = False
    timeout: int = 10

class BulkEmailRequest(BaseModel):
    emails: List[str]
    check_smtp: bool = False
    timeout: int = 5

class ValidationResponse(BaseModel):
    email: str
    is_valid: bool
    domain_exists: bool
    mx_records: List[str]
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    smtp_valid: Optional[bool] = None
    validation_time: float
    timestamp: str

class BulkValidationResponse(BaseModel):
    total: int
    valid: int
    invalid: int
    processing_time: float
    results: List[ValidationResponse]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Email Validator Core Logic
@dataclass
class ValidationResult:
    is_valid: bool
    email: str
    errors: list
    warnings: list
    domain_exists: bool
    mx_records: list
    smtp_valid: Optional[bool] = None

class EmailValidator:
    def __init__(self):
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        self.domain_corrections = {
            'gmail.co': 'gmail.com',
            'gmail.cm': 'gmail.com',
            'gmial.com': 'gmail.com',
            'gmai.com': 'gmail.com',
            'yahoo.co': 'yahoo.com',
            'yahoo.cm': 'yahoo.com',
            'hotmail.co': 'hotmail.com',
            'hotmail.cm': 'hotmail.com',
            'outlook.co': 'outlook.com',
            'outlok.com': 'outlook.com',
        }
    
    async def validate_email(self, email: str, check_smtp: bool = False, timeout: int = 10) -> ValidationResult:
        """Validate email with comprehensive checks"""
        start_time = asyncio.get_event_loop().time()
        
        result = ValidationResult(
            is_valid=True,
            email=email.lower().strip(),
            errors=[],
            warnings=[],
            domain_exists=False,
            mx_records=[]
        )
        
        # Format validation
        if not self._validate_format(result.email):
            result.errors.append("Invalid email format")
            result.is_valid = False
        
        # Extract domain
        try:
            local, domain = result.email.split('@')
            if not local or not domain:
                result.errors.append("Missing local or domain part")
                result.is_valid = False
                return result
        except ValueError:
            result.errors.append("Invalid email structure")
            result.is_valid = False
            return result
        
        # Domain validation
        try:
            domain_valid, mx_records = await self._validate_domain_async(domain, timeout)
            result.domain_exists = domain_valid
            result.mx_records = mx_records
            
            if not domain_valid:
                result.errors.append(f"Domain '{domain}' does not exist or has no MX records")
                result.is_valid = False
        except Exception as e:
            result.errors.append(f"DNS lookup failed: {str(e)}")
            result.is_valid = False
        
        # SMTP validation
        if check_smtp and result.domain_exists and result.mx_records:
            try:
                smtp_result = await self._validate_smtp_async(result.email, result.mx_records[0], timeout)
                result.smtp_valid = smtp_result
                
                if smtp_result is False:
                    result.warnings.append("SMTP server rejected the email address")
                elif smtp_result is None:
                    result.warnings.append("Could not verify with SMTP server")
            except Exception as e:
                result.warnings.append(f"SMTP validation failed: {str(e)}")
        
        return result
    
    def _validate_format(self, email: str) -> bool:
        """Validate email format"""
        if not email or len(email) > 254:
            return False
        
        if not self.email_pattern.match(email):
            return False
        
        try:
            local, domain = email.split('@')
            
            # Local part checks
            if len(local) > 64 or not local:
                return False
            
            if local.startswith('.') or local.endswith('.') or '..' in local:
                return False
            
            # Domain checks
            if len(domain) > 253 or not domain:
                return False
            
            return True
        except:
            return False
    
    async def _validate_domain_async(self, domain: str, timeout: int) -> tuple:
        """Async domain validation"""
        def dns_lookup():
            try:
                # Try MX records first
                mx_answers = dns.resolver.resolve(domain, 'MX')
                mx_records = [str(mx.exchange).rstrip('.') for mx in sorted(mx_answers, key=lambda x: x.preference)]
                return True, mx_records
            except dns.resolver.NXDOMAIN:
                return False, []
            except dns.resolver.NoAnswer:
                # Try A record
                try:
                    dns.resolver.resolve(domain, 'A')
                    return True, [domain]
                except:
                    return False, []
            except:
                return False, []
        
        return await asyncio.get_event_loop().run_in_executor(None, dns_lookup)
    
    async def _validate_smtp_async(self, email: str, mx_server: str, timeout: int) -> Optional[bool]:
        """Async SMTP validation"""
        def smtp_check():
            try:
                server = smtplib.SMTP(timeout=timeout)
                server.connect(mx_server, 25)
                server.helo('localhost')
                server.mail('test@example.com')
                
                code, message = server.rcpt(email)
                server.quit()
                
                if code == 250:
                    return True
                elif code in [550, 551, 553]:
                    return False
                else:
                    return None
            except:
                return None
        
        return await asyncio.get_event_loop().run_in_executor(None, smtp_check)
    
    def get_suggestions(self, email: str) -> List[str]:
        """Get typo suggestions"""
        suggestions = []
        
        if '@' not in email:
            return suggestions
        
        try:
            local, domain = email.split('@')
            
            if domain in self.domain_corrections:
                suggestions.append(f"{local}@{self.domain_corrections[domain]}")
        except:
            pass
        
        return suggestions