# Book Detection and OCR API

This project provides a FastAPI-based backend for detecting books in images and extracting text using a pre-trained Faster R-CNN model and Tesseract OCR. The API accepts images, processes them to detect books, crops the detected regions, and performs OCR to extract text.

## Features

- Detect books in images using a pre-trained Faster R-CNN model.
- Extract text from detected book regions using Tesseract OCR.
- Return the cropped book images and extracted text in the response.

## Requirements

- Python 3.8 or higher
- Torch
- torchvision
- FastAPI
- Uvicorn
- Pillow
- Pytesseract
- Numpy
- Opencv-python-headless

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/laithml/BookScannerAPI.git
    cd BookScannerAPI
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure Tesseract is installed on your system:
    - **macOS**: Install using Homebrew
        ```sh
        brew install tesseract
        ```
    - **Linux**: Install using apt-get
        ```sh
        sudo apt-get install tesseract-ocr
        ```
    - **Windows**: Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and run it.

5. Update Tesseract path in the code if necessary. For macOS, the path is set in the `book_detection_model.py` file:
    ```python
    pytesseract.pytesseract_cmd = r'/opt/homebrew/bin/tesseract'
    ```

6. Place the pre-trained model `final_model_1.pth` in the project directory.

## Usage

1. Run the FastAPI server:
    ```sh
    uvicorn main:app --reload
    ```

2. The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

### Upload and Process Images

- **URL**: `/upload-images`
- **Method**: `POST`
- **Description**: Upload images for book detection and OCR processing.
- **Request Body**:
    - `files`: List of image files (PNG or JPEG).
- **Response**:
    - `books`: List of detected books with bounding boxes, cropped images (base64-encoded), and extracted text.

#### Example Request

```sh
curl -X POST "http://127.0.0.1:8000/upload-images" -F "files=@path/to/your/image1.jpg" -F "files=@path/to/your/image2.png"
