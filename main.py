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

# -------------------------
# ROOT ROUTE
# -------------------------
@app.get("/")
def home():
    return {"message": "Flood Detection API is running successfully!"}


# -------------------------
# Models
# -------------------------
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


# -------------------------
# JSON Parser Utility
# -------------------------
def parse_gemini_response(text: str):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return {
            "risk_level": "Medium",
            "description": "Default analysis",
            "recommendations": ["Stay cautious"],
            "elevation": 50.0,
            "distance_from_water": 1000.0
        }

    data = json.loads(match.group())

    # Ensure numeric values
    elevation = data.get("elevation", 50)
    distance = data.get("distance_from_water", 1000)

    try:
        elevation = float(elevation)
    except:
        elevation = random.uniform(5, 150)

    try:
        distance = float(distance)
    except:
        distance = random.uniform(100, 2000)

    return {
        "risk_level": data.get("risk_level", "Medium"),
        "description": data.get("description", ""),
        "recommendations": data.get("recommendations", []),
        "elevation": elevation,
        "distance_from_water": distance
    }


# -------------------------
# COORDINATE ANALYSIS
# -------------------------
@app.post("/api/analyze/coordinates")
async def analyze_coordinates(coords: CoordinateRequest):

    logger.info(f"Analyze coords: {coords.latitude}, {coords.longitude}")

    prompt = f"""
    You are an expert hydrologist.

    Analyze flood risk for:
    Latitude: {coords.latitude}
    Longitude: {coords.longitude}

    IMPORTANT RULES:
    - Respond ONLY in strict JSON.
    - "elevation" MUST be a pure number (float). No text, no units.
    - "distance_from_water" MUST be a pure number (float). No text, no words.
    - If unsure, make your BEST numeric estimation.
    - DO NOT output strings like "unknown" or "not applicable".
    - DO NOT include units like "m" or "meters".
    - DO NOT include any extra text outside the JSON.

    JSON FORMAT:
    {{
        "risk_level": "Low/Medium/High/Very High",
        "description": "text",
        "recommendations": ["text1", "text2"],
        "elevation": 12.5,
        "distance_from_water": 340.2
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

        return {
            "success": True,
            "risk_level": random.choice(["Low", "Medium", "High"]),
            "description": "Simulated coordinate analysis",
            "recommendations": ["Check flood zone maps"],
            "elevation": random.uniform(10, 100),
            "distance_from_water": random.uniform(200, 2000),
            "message": "Fallback returned (AI failed)"
        }


# -------------------------
# IMAGE ANALYSIS
# -------------------------
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
    You are an expert hydrologist.

    Analyze the FLOOD RISK from this image.

    IMPORTANT RULES:
    - Respond ONLY in strict JSON.
    - "elevation" MUST be a number only.
    - "distance_from_water" MUST be a number only.
    - No text values allowed for numeric fields.
    - If unsure, estimate the most realistic numbers.

    JSON FORMAT:
    {
        "risk_level": "Low/Medium/High/Very High",
        "description": "text",
        "recommendations": ["text1", "text2"],
        "elevation": 10.5,
        "distance_from_water": 250.0
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
            "recommendations": ["Monitor weather"],
            "elevation": 50.0,
            "distance_from_water": 1000.0,
            "message": "Fallback used"
        }


# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
