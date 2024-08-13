# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to reduce buffer flushing and ensure consistent logging output
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install dependencies required for Tesseract OCR and other necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Make sure the model file is correctly copied into the image
COPY final_model_1.pth /app/final_model_1.pth

# Expose port 8000 to allow external access
EXPOSE 8000

# Use 'gunicorn' with Uvicorn worker for better performance in production
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
