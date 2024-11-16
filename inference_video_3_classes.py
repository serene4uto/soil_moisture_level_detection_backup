import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from data_loader import *
from create_model import *
from PIL import Image
import cv2

class Inference:
    def __init__(self, model_path, num_classes=5, img_size=(224, 224)):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = clc_model('mobilenetv3_large_100', num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, video_path, class_names):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img).unsqueeze(0)
            img = img.to(self.device)

            with torch.no_grad():
                outputs = self.model(img)
                _, predicted = torch.max(outputs, 1)

            predicted_class = class_names[predicted.item()]

            cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Video Prediction', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == "__main__":
    model_path = r'C:\Users\kde10\OneDrive\바탕 화면\MoistLevelUpdate_20241111\update_20241111\mobilenetv3_3_classes.pth'
    video_path = 0

    class_names = ['high', 'low', 'medium']

    tester = Inference(model_path, num_classes=len(class_names))

    predicted_class = tester.predict(video_path, class_names)
