# book_detection_model.py
import base64
import io
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from fastapi import HTTPException
from pytesseract import pytesseract
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

# Update Tesseract path for macOS
pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


def get_transform(train=True):
    transforms_list = [
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if train:
        transforms_list.insert(0, transforms.RandomHorizontalFlip(0.5))
        transforms_list.insert(0, transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        transforms_list.insert(0, transforms.RandomRotation(15))
    return transforms.Compose(transforms_list)


class ExpandedFastRCNNPredictor(FastRCNNPredictor):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
        hidden_layer_size = 1024
        self.cls_score = nn.Sequential(
            nn.Linear(in_channels, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_classes)
        )
        self.bbox_pred = nn.Sequential(
            nn.Linear(in_channels, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_classes * 4)
        )


class CNNModel:
    def __init__(self, lr=0.001):
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        num_classes = 2  # 1 class (book) + background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = ExpandedFastRCNNPredictor(in_features, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=False)

    def predict(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([image_tensor])
        return prediction


def extract_texts_from_boxes(frame, boxes, scores, threshold=0.5, lang='eng+ara+heb'):
    texts = []
    h, w, _ = frame.shape
    for box, score in zip(boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x1 < x2 and y1 < y2:
                cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
                try:
                    pil_image = Image.fromarray(cropped_image)
                    text = pytesseract.image_to_string(pil_image, lang=lang)
                    texts.append({'box': (int(x1), int(y1), int(x2), int(y2)), 'text': text})
                except Exception as e:
                    print(f"Error processing image for OCR: {e}")
    return texts


def process_image(image_tensor, original_image, model):
    try:
        original_width, original_height = original_image.size
        image_tensor = image_tensor.to(model.device)
        predictions = model.predict(image_tensor)
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Scaling the bounding boxes back to the original image size
        scale_x = original_width / 800
        scale_y = original_height / 800
        pred_boxes = pred_boxes * [scale_x, scale_y, scale_x, scale_y]

        extracted_texts = extract_texts_from_boxes(np.array(original_image), pred_boxes, scores)

        books_info = []
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_image = original_image.crop((x1, y1, x2, y2))
            buffered = io.BytesIO()
            cropped_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            text = pytesseract.image_to_string(cropped_image)
            books_info.append({
                "box": [x1, y1, x2, y2],
                "image": img_str,
                "text": text
            })

        return books_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
