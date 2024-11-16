import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from data_loader import *
from create_model import *
from PIL import Image


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

    def predict(self, image_path, class_names):
        img = Image.open(image_path).convert('RGB')  
        img = self.transform(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)

        predicted_class = class_names[predicted.item()]
        return predicted_class


if __name__ == "__main__":
    model_path = r'C:\Users\rnd2\Desktop\soil_moisture_detection_demo\soil_moisture_level_detection_backup\mobilenetv3_3_classes.pth'

    # C:\Users\kde10\OneDrive\바탕 화면\MoistLevelUpdate_20241111\update_20241111\test_set\low
    image_path = r"C:\Users\rnd2\Desktop\soil_moisture_detection_demo\soil_moisture_level_detection_backup\test_set\high\240924_15_00_23_#116.bmp"

    class_names = ['high', 'low', 'medium']

    tester = Inference(model_path, num_classes=len(class_names))

    predicted_class = tester.predict(image_path, class_names)
    print(f'Predicted Class: {predicted_class}')
