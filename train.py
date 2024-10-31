import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from data_loader import *
from create_model import *


class MobileNetTrainer:
    def __init__(self, train_loader, val_loader, num_classes=11, lr=0.001, num_epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = clc_model('mobilenetv3_large_100', num_classes)
        self.model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2, verbose=True)

        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(self.train_loader)
            train_acc = 100 * correct / total

            val_loss, val_acc = self.validate()

            self.scheduler.step(val_loss)

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(self.val_loader)
        val_acc = 100 * val_correct / val_total
        return val_loss, val_acc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    train_dir = r'D:\dataset\all\new_split\train'
    val_dir = r'D:\dataset\all\new_split\test'
    batch_size = 32
    num_classes = 6
    num_epochs = 10
    learning_rate = 0.001

    data_loader = DataLoaderWrapper(train_dir=train_dir, val_dir=val_dir, batch_size=batch_size)
    train_loader, val_loader = data_loader.get_loaders()

    trainer = MobileNetTrainer(train_loader, val_loader, num_classes=num_classes, lr=learning_rate, num_epochs=num_epochs)

    trainer.train()

    trainer.save_model('mobilenetv3_6_classes.pth')
