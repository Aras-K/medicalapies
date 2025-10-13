# ========================================
# main.py - FastAPI X-Ray Analysis API
# ========================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import secrets
import logging
import io
import pydicom
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import base64

# Import your analyzer
from torchxrayvision_analyzer import analyze_with_ai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="X-Ray Analysis API",
    description="Medical X-Ray Analysis using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBasic()

# ========================================
# USER MANAGEMENT (No Database)
# ========================================

USERS = {
    "client_test": hashlib.sha256("TestPass2025!".encode()).hexdigest(),
    "demo_doctor": hashlib.sha256("DemoPass123!".encode()).hexdigest(),
    "medical_api": hashlib.sha256("MedicalAPI2025!".encode()).hexdigest(),
}

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Verify user credentials"""
    password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
    
    if credentials.username not in USERS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    if USERS[credentials.username] != password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username

# ========================================
# ENDPOINTS
# ========================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - API documentation"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>X-Ray Analysis API</title>
        <style>
            body { 
                font-family: -apple-system, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            .status { 
                background: #48bb78; 
                color: white; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center;
                font-size: 1.2em;
            }
            .endpoint {
                background: #f7fafc;
                padding: 20px;
                margin: 20px 0;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            code {
                background: #2d3748;
                color: #48bb78;
                padding: 2px 8px;
                border-radius: 4px;
            }
            h1 { color: #2d3748; }
            .test-accounts {
                background: #edf2f7;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid #e2e8f0;
            }
            th { background: #f7fafc; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 X-Ray Analysis API (FastAPI)</h1>
            
            <div class="status">
                ✅ API is Online and Ready
            </div>
            
            <h2>API Endpoints</h2>
            
            <div class="endpoint">
                <strong>POST /api/analyze-xray</strong>
                <p>Upload DICOM file for AI analysis</p>
                <p>Authentication: Basic Auth required</p>
            </div>
            
            <div class="endpoint">
                <strong>GET /health</strong>
                <p>Health check endpoint</p>
            </div>
            
            <div class="endpoint">
                <strong>GET /docs</strong>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "X-Ray Analysis API",
        "version": "1.0.0"
    }

@app.post("/api/analyze-xray")
async def analyze_xray(
    file: UploadFile = File(...),
    username: str = Depends(verify_credentials)
):
    """
    Analyze X-ray DICOM file
    
    - **file**: DICOM file upload
    - **Authentication**: Basic Auth required
    """
    
    try:
        # Read file
        contents = await file.read()
        logger.info(f"Processing file: {file.filename} ({len(contents)} bytes) for user: {username}")
        
        # Validate file size (max 50MB)
        max_size = 50 * 1024 * 1024
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {max_size // 1024 // 1024}MB"
            )
        
        # Parse DICOM
        try:
            dicom = pydicom.dcmread(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid DICOM file: {str(e)}"
            )
        
        # Extract metadata
        metadata = {
            "patient_id": str(getattr(dicom, 'PatientID', 'ANONYMOUS')),
            "study_date": str(getattr(dicom, 'StudyDate', 'Unknown')),
            "modality": str(getattr(dicom, 'Modality', 'Unknown')),
            "body_part": str(getattr(dicom, 'BodyPartExamined', 'CHEST')),
            "view_position": str(getattr(dicom, 'ViewPosition', 'Unknown')),
            "image_size": [
                int(getattr(dicom, 'Rows', 0)),
                int(getattr(dicom, 'Columns', 0))
            ]
        }
        
        # Get pixel array
        try:
            pixel_array = dicom.pixel_array
        except Exception as e:
            logger.error(f"Could not extract pixel data: {e}")
            raise HTTPException(
                status_code=422,
                detail="Could not extract image data from DICOM. Please check the file."
            )
        
        # Run AI analysis
        logger.info("Running AI analysis...")
        ai_results = analyze_with_ai(pixel_array, metadata)
        
        # Build response
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "username": username,
            "file": {
                "name": file.filename,
                "size": len(contents),
                "content_type": file.content_type
            },
            "metadata": metadata,
            "analysis": ai_results
        }
        
        logger.info(f"Analysis complete for {username}")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/api/analyze-base64")
async def analyze_base64(
    data: dict,
    username: str = Depends(verify_credentials)
):
    """
    Analyze X-ray from base64 encoded DICOM
    
    Body should contain:
    - dicom_base64: Base64 encoded DICOM file
    """
    
    try:
        if "dicom_base64" not in data:
            raise HTTPException(
                status_code=400,
                detail="Missing 'dicom_base64' in request body"
            )
        
        # Decode base64
        try:
            dicom_bytes = base64.b64decode(data["dicom_base64"])
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid base64 encoding"
            )
        
        # Parse DICOM
        dicom = pydicom.dcmread(io.BytesIO(dicom_bytes))
        
        # Rest is same as file upload...
        metadata = {
            "patient_id": str(getattr(dicom, 'PatientID', 'ANONYMOUS')),
            "study_date": str(getattr(dicom, 'StudyDate', 'Unknown')),
            "modality": str(getattr(dicom, 'Modality', 'Unknown')),
            "body_part": str(getattr(dicom, 'BodyPartExamined', 'CHEST')),
            "view_position": str(getattr(dicom, 'ViewPosition', 'Unknown')),
        }
        
        pixel_array = dicom.pixel_array
        ai_results = analyze_with_ai(pixel_array, metadata)
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "username": username,
            "metadata": metadata,
            "analysis": ai_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)