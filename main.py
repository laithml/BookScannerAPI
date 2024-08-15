from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List
from book_detection_model import (CNNModel, process_image, get_transform)
from PIL import Image
import uvicorn
import io
import requests

app = FastAPI()

model = CNNModel()
model.load_model("final_model_1.pth")
transform = get_transform(train=False)

@app.get("/")
async def root():
    print("GET request to the root endpoint")
    return {"message": "Welcome to the book detection API"}

@app.post("/upload-image-url")
async def upload_image_url(imageUrl: str):
    print(f"POST request to the upload-image-url endpoint with URL: {imageUrl}")
    try:
        # Download the image from the URL
        response = requests.get(imageUrl)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Unable to download image from URL.")

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image_tensor = transform(image)
        books_info = process_image(image_tensor, image, model)

        return JSONResponse(content={"books": books_info})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
