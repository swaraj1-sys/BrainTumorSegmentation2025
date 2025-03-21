from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# Load the trained model
model = load_model("done_model.h5")  # Ensure this file is in the same directory

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))  # Change size if needed
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")  # Convert to RGB
    image = preprocess_image(image)

    prediction = model.predict(image)
    mask_present = np.any(prediction > 0.5)  # Check if mask pixels exist

    return {"mask_detected": bool(mask_present)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
