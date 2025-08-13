from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import os
import asyncio
from datetime import datetime
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import io
import json
import re
import random
from PIL import Image as PILImage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(
    title="Flood Detection API",
    description="Advanced flood risk assessment using Gemini AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CoordinateRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude between -90 and 90")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude between -180 and 180")

class AnalysisResponse(BaseModel):
    success: bool
    risk_level: str
    description: str
    recommendations: List[str]
    elevation: float
    distance_from_water: float
    ai_analysis: Optional[str] = None
    message: str

def parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini AI response and extract structured data"""
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_data = json.loads(json_str)
            return {
                "risk_level": parsed_data.get("risk_level", "Medium"),
                "description": parsed_data.get("description", "Analysis completed"),
                "recommendations": parsed_data.get("recommendations", []),
                "elevation": parsed_data.get("elevation", 50.0),
                "distance_from_water": parsed_data.get("distance_from_water", 1000.0),
                "analysis": parsed_data.get("analysis", response_text)
            }
        return {
            "risk_level": "Medium",
            "description": "Analysis completed",
            "recommendations": ["Monitor weather conditions", "Stay informed about local alerts"],
            "elevation": 50.0,
            "distance_from_water": 1000.0,
            "analysis": response_text
        }
    except Exception as e:
        logger.error(f"Error parsing Gemini response: {str(e)}")
        return generate_fallback_response()

def generate_fallback_response() -> dict:
    """Generate fallback response when analysis fails"""
    return {
        "risk_level": "Medium",
        "description": "Analysis completed with default values",
        "recommendations": [
            "Monitor local weather reports",
            "Check flood risk maps regularly",
            "Prepare emergency supplies"
        ],
        "elevation": 50.0,
        "distance_from_water": 1000.0,
        "analysis": "Default analysis provided"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Flood Detection API with Gemini AI",
        "version": "2.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health_check": "/health",
            "image_analysis": "/api/analyze/image",
            "coordinate_analysis": "/api/analyze/coordinates",
            "documentation": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "ai_model": "Gemini 2.0 Flash",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.post("/api/analyze/coordinates")
async def analyze_coordinates(coords: CoordinateRequest):
    """
    Analyze flood risk based on geographic coordinates
    
    Parameters:
    - latitude: Between -90 and 90
    - longitude: Between -180 and 180
    
    Returns comprehensive flood risk assessment
    """
    try:
        logger.info(f"Analyzing coordinates: {coords.latitude}, {coords.longitude}")
        
        prompt = f"""
        Analyze flood risk for location at latitude {coords.latitude}, longitude {coords.longitude}.
        
        Provide detailed assessment including:
        1. Risk Level (Low/Medium/High/Very High)
        2. Description of risk factors
        3. 3-5 specific recommendations
        4. Estimated elevation in meters
        5. Estimated distance from nearest water body in meters
        6. Detailed analysis of terrain and flood risk factors
        
        Format response as JSON with these fields:
        - risk_level
        - description
        - recommendations (array)
        - elevation (number)
        - distance_from_water (number)
        - analysis (detailed text)
        """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            parsed_data = parse_gemini_response(response.text)
            
            return {
                "success": True,
                "risk_level": parsed_data["risk_level"],
                "description": parsed_data["description"],
                "recommendations": parsed_data["recommendations"],
                "elevation": parsed_data["elevation"],
                "distance_from_water": parsed_data["distance_from_water"],
                "ai_analysis": parsed_data["analysis"],
                "message": "Coordinate analysis completed successfully"
            }
            
        except Exception as ai_error:
            logger.error(f"Gemini AI error: {str(ai_error)}")
            fallback = generate_coordinate_fallback(coords.latitude, coords.longitude)
            return {
                "success": True,
                "message": "Analysis completed with simulated data",
                **fallback
            }
            
    except Exception as e:
        logger.error(f"Coordinate analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze flood risk from terrain image
    
    Parameters:
    - file: Image file (JPEG, PNG) under 10MB
    
    Returns flood risk assessment with visual analysis
    """
    try:
        logger.info(f"Image analysis request: {file.filename}")
        
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are accepted")
        
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        
        # Process image
        image_data = await file.read()
        try:
            image = PILImage.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as img_error:
            logger.error(f"Image processing error: {str(img_error)}")
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Generate analysis prompt
        prompt = """
        Analyze this terrain image for flood risk assessment.
        Provide detailed assessment including:
        1. Risk Level (Low/Medium/High/Very High)
        2. Description of visible risk factors
        3. 3-5 specific recommendations
        4. Estimated elevation in meters
        5. Estimated distance from visible water bodies
        6. Detailed analysis of what you observe
        
        Format response as JSON with these fields:
        - risk_level
        - description
        - recommendations (array)
        - elevation (number)
        - distance_from_water (number)
        - analysis (detailed text)
        """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([prompt, image])
            parsed_data = parse_gemini_response(response.text)
            
            return {
                "success": True,
                "risk_level": parsed_data["risk_level"],
                "description": parsed_data["description"],
                "recommendations": parsed_data["recommendations"],
                "elevation": parsed_data["elevation"],
                "distance_from_water": parsed_data["distance_from_water"],
                "ai_analysis": parsed_data["analysis"],
                "message": "Image analysis completed successfully"
            }
            
        except Exception as ai_error:
            logger.error(f"Gemini AI error: {str(ai_error)}")
            fallback = generate_image_fallback()
            return {
                "success": True,
                "message": "Analysis completed with simulated data",
                **fallback
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

# Helper functions
def generate_coordinate_fallback(lat: float, lng: float) -> dict:
    """Generate simulated coordinate analysis"""
    risk_level = "Medium"
    if lat > 40 and lng < -70:
        risk_level = "Low"
    elif lat < 30 and lng > -90:
        risk_level = "High"
    
    return {
        "risk_level": risk_level,
        "description": f"Simulated analysis for coordinates {lat}, {lng}",
        "recommendations": [
            "Check local flood maps",
            "Monitor weather forecasts",
            "Prepare emergency plan"
        ],
        "elevation": 50.0,
        "distance_from_water": 1000.0,
        "analysis": "Simulated analysis data provided"
    }

def generate_image_fallback() -> dict:
    """Generate simulated image analysis"""
    return {
        "risk_level": "Medium",
        "description": "Simulated image analysis results",
        "recommendations": [
            "Review terrain carefully",
            "Consult local experts",
            "Verify with additional data"
        ],
        "elevation": 50.0,
        "distance_from_water": 1000.0,
        "analysis": "Simulated image analysis data provided"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
    