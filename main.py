# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from book_detection_model import (CNNModel, process_image, get_transform)
from PIL import Image
import uvicorn
import io

app = FastAPI()

model = CNNModel()
model.load_model("final_model_1.pth")
transform = get_transform(train=False)


@app.post("/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    try:
        results = []
        for idx, file in enumerate(files):
            content_type = file.content_type
            filename = file.filename
            if content_type not in ["image/png", "image/jpeg"]:
                raise HTTPException(status_code=400, detail="Invalid file type. Only PNG and JPEG are supported.")

            image = await file.read()
            image = Image.open(io.BytesIO(image)).convert("RGB")
            image_tensor = transform(image)
            books_info = process_image(image_tensor, image, model)
            results.extend(books_info)
        return JSONResponse(content={"books": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
