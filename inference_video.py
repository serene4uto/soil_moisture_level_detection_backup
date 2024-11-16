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
import logging
from datetime import datetime

class Inference:
    def __init__(self, model_path, num_classes=5, img_size=(224, 224), log_file='prediction.log'):
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
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

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
            logging.info(f'Predicted: {predicted_class}')
            cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = r'C:\Users\rnd2\Desktop\soil_moistrure_demo\soil_moisture_level_detection_backup\mobilenetv3_6_classes.pth'
    video_path = 2
    class_names = ['Excess_moisture_100', 'Excess_moisture_80', 'Insufficient_moisture_0', 'Insufficient_moisture_20', 'Optimal_moisture_40', 'Optimal_moisture_60']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'prediction_{timestamp}.log'
    
    tester = Inference(model_path, num_classes=len(class_names), log_file=log_file)
    tester.predict(video_path, class_names)
