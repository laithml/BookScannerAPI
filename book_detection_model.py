import base64
import io

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from fastapi import HTTPException
from pytesseract import pytesseract
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms  # Import NMS
import os
import pytesseract

# Update Tesseract path
pytesseract.tesseract_cmd = r'/usr/bin/tesseract'



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


def process_image(image_tensor, original_image, model):
    try:
        original_width, original_height = original_image.size
        image_tensor = image_tensor.to(model.device)
        predictions = model.predict(image_tensor)
        pred_boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']

        # Apply NMS
        keep = nms(pred_boxes, scores, 0.3)
        pred_boxes = pred_boxes[keep].cpu().numpy()

        # Scaling the bounding boxes back to the original image size
        scale_x = original_width / 800
        scale_y = original_height / 800
        pred_boxes = pred_boxes * [scale_x, scale_y, scale_x, scale_y]

        books_info = []
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_image = original_image.crop((x1, y1, x2, y2))
            buffered = io.BytesIO()
            cropped_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            text = pytesseract.image_to_string(cropped_image, lang='eng+ara+heb')
            books_info.append({
                "box": [x1, y1, x2, y2],
                "image": img_str,
                "text": text
            })

        return books_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
