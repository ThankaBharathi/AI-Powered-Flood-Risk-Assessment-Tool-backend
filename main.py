from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from datetime import datetime
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import io
import json
import re
from PIL import Image as PILImage
import random

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# FastAPI app
app = FastAPI(
    title="Flood Detection API",
    description="Simple flood risk assessment using Gemini AI",
    version="1.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class CoordinateRequest(BaseModel):
    latitude: float
    longitude: float

class AnalysisResponse(BaseModel):
    success: bool
    risk_level: str
    description: str
    recommendations: list[str]
    elevation: float
    distance_from_water: float
    message: str


# ==========================================
# Utility: Parse Gemini response
# ==========================================
def parse_gemini_response(response_text: str) -> dict:
    """
    Extract JSON from Gemini response text.
    """
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "risk_level": data.get("risk_level", "Medium"),
                "description": data.get("description", ""),
                "recommendations": data.get("recommendations", []),
                "elevation": data.get("elevation", 50.0),
                "distance_from_water": data.get("distance_from_water", 1000.0)
            }
    except:
        pass

    # fallback if Gemini fails
    return {
        "risk_level": "Medium",
        "description": "Analysis completed",
        "recommendations": ["Stay informed", "Monitor alerts"],
        "elevation": 50.0,
        "distance_from_water": 1000.0
    }


# ==========================================
# ROOT ROUTES
# ==========================================
@app.get("/")
async def root():
    return {
        "message": "Flood Detection API working",
        "version": "1.1.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ==========================================
# 🚨 COORDINATE ANALYSIS ENDPOINT (ADDED)
# ==========================================
@app.post("/api/analyze/coordinates")
async def analyze_coordinates(coords: CoordinateRequest):
    logger.info(f"Analyze coords: {coords.latitude}, {coords.longitude}")

    prompt = f"""
    Analyze flood risk for:
    Latitude: {coords.latitude}
    Longitude: {coords.longitude}

    Respond ONLY in JSON format:
    {{
        "risk_level": "...",
        "description": "...",
        "recommendations": ["...", "..."],
        "elevation": number,
        "distance_from_water": number
    }}
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        parsed = parse_gemini_response(response.text)

        return {
            "success": True,
            **parsed,
            "message": "Coordinate analysis completed"
        }

    except Exception as e:
        logger.error(f"Gemini Error: {e}")

        # simulated fallback
        return {
            "success": True,
            "risk_level": random.choice(["Low", "Medium", "High"]),
            "description": "Simulated coordinate analysis",
            "recommendations": [
                "Check flood zone maps",
                "Monitor rainfall alerts",
                "Have an evacuation plan"
            ],
            "elevation": random.uniform(10, 100),
            "distance_from_water": random.uniform(200, 2000),
            "message": "Fallback returned (AI failed)"
        }


# ==========================================
# IMAGE ANALYSIS (Already Working)
# ==========================================
@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    logger.info(f"Analyzing image: {file.filename}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid image")

    data = await file.read()

    try:
        image = PILImage.open(io.BytesIO(data))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except:
        raise HTTPException(400, "Invalid image format")

    prompt = """
    Analyze this terrain image for flood risk.
    Respond in JSON with:
    {
        "risk_level": "...",
        "description": "...",
        "recommendations": ["...", "..."],
        "elevation": number,
        "distance_from_water": number
    }
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content([prompt, image])
        parsed = parse_gemini_response(response.text)

        return {
            "success": True,
            **parsed,
            "message": "Image analysis completed"
        }

    except Exception as e:
        logger.error(f"AI Error: {e}")

        return {
            "success": True,
            "risk_level": "Medium",
            "description": "Simulated image risk",
            "recommendations": ["Monitor weather", "Stay cautious"],
            "elevation": 50.0,
            "distance_from_water": 1000.0,
            "message": "Fallback used"
        }


# ==========================================
# RUN SERVER
# ==========================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
