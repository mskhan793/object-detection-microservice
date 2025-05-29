from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List
import os
import uuid

# Create FastAPI instance
app = FastAPI(
    title="Object Detection API",
    description="A microservice for detecting objects in images using YOLOv8",
    version="1.0.0"
)

# Define response model for structured output
class DetectionResult(BaseModel):
    boxes: List[List[float]]
    labels: List[str]
    scores: List[float]

# Directory for storing output images
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the YOLOv8 model
model = None

@app.on_event("startup")
async def load_model():
    """Load the YOLOv8 model at application startup"""
    global model
    try:
        # Download model if it doesn't exist
        model = YOLO("yolov8n.pt")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Continue without failing, will attempt to download on first request

@app.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Object Detection API is running!",
        "usage": "POST an image to /detect to perform object detection",
        "docs": "Visit /docs for the interactive API documentation"
    }

@app.post("/detect", response_model=DetectionResult)
async def detect_objects(file: UploadFile = File(...), 
                         save_image: bool = True,
                         confidence: float = 0.25):
    """
    Detect objects in an uploaded image
    
    Parameters:
    - file: The image file to analyze
    - save_image: Whether to save the output image with boxes (default: True)
    - confidence: Minimum confidence threshold for detections (default: 0.25)
    
    Returns:
    - JSON with bounding boxes, labels and confidence scores
    """
    global model
    
    # Check if model is loaded
    if model is None:
        try:
            model = YOLO("yolov8n.pt")
            print("Model loaded on first request")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        # Read and decode the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run YOLOv8 inference with the specified confidence threshold
        results = model.predict(img, conf=confidence, device=0)  # device=0 to use GPU
        
        # Extract predictions
        predictions = results[0]
        boxes = predictions.boxes.xyxy.cpu().numpy().tolist()
        scores = predictions.boxes.conf.cpu().numpy().tolist()
        labels = [model.names[int(cls)] for cls in predictions.boxes.cls.cpu().numpy()]
        
        # Draw bounding boxes on the image
        output_img = img.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{labels[i]}: {scores[i]:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Save the output image with a unique filename
        output_filename = None
        if save_image:
            unique_id = str(uuid.uuid4())[:8]
            output_filename = f"{OUTPUT_DIR}/detection_{unique_id}.jpg"
            cv2.imwrite(output_filename, output_img)
            print(f"Saved output image to {output_filename}")
            
        # Return detection results
        return DetectionResult(boxes=boxes, labels=labels, scores=scores)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/output/{image_name}")
async def get_output_image(image_name: str):
    """Retrieve a processed output image by name"""
    image_path = f"{OUTPUT_DIR}/{image_name}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)
