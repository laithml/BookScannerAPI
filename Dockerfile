# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to reduce buffer flushing and ensure consistent logging output
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr

# Copy only the necessary files first to leverage Docker cache
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt, including pytesseract
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Copy the model file into the image (if not already copied by the previous command)
COPY final_model_1.pth /app/final_model_1.pth

# Expose port 5000 to allow external access
EXPOSE 5000

# Run main.py when the container launches
CMD ["python", "main.py"]
