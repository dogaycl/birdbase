from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uvicorn
from io import BytesIO

from core.inference import YOLOv8ONNX
from core.species_info import get_species_info

app = FastAPI(title="BirdBase API", description="AI Backend for bird detection and info.")

# Allow CORS for Web Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_classes():
    classes_path = "../ai_model/data/CUB_200_2011/CUB_200_2011/classes.txt"
    try:
        with open(classes_path, "r") as f:
            return [line.strip().split(".", 1)[1].replace("_", " ") for line in f]
    except Exception:
        return ['Eagle', 'Hawk', 'Sparrow', 'Pigeon', 'Owl']

CLASSES = load_classes()
# Initialize ONNX inference class. Point to the expected path of the exported model.
MODEL_PATH = "../ai_model/weights/best_train3_pack.onnx" 
detector = YOLOv8ONNX(MODEL_PATH, CLASSES)

@app.get("/")
def read_root():
    return {"message": "Welcome to the BirdBase API"}

@app.post("/predict/")
async def predict_bird(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail=f"Ensure the file is an image. Received type: {file.content_type}")

    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file or cannot be decoded.")

    # Run inference with a standard threshold now that we have a fully trained model.
    predictions = detector.predict(img, conf_threshold=0.3)
    
    # Check if a bird was detected
    if len(predictions) == 0:
        return {"detected": False, "message": "No bird found with confidence > 0.3. Try a clearer image."}
    
    # Top prediction
    top_pred = predictions[0]
    species_name = top_pred['class']
    
    # Enrich with species information
    details = get_species_info(species_name)
    
    return {
        "detected": True,
        "species": species_name,
        "confidence": top_pred['confidence'],
        "bounding_box": top_pred['bbox'],
        "info": details
    }

@app.get("/species/{name}")
def get_species(name: str):
    info = get_species_info(name)
    if "error" in info:
        raise HTTPException(status_code=404, detail=info["error"])
    return info

if __name__ == "__main__":
    print("[*] Starting BirdBase Backend...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
