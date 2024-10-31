import os
import shutil
import random

def split_dataset(source_dir, train_dir, test_dir, train_ratio=0.7):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    

    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)
        
        if os.path.isdir(class_path):

            images = os.listdir(class_path)
            random.shuffle(images)  
            
            split_index = int(len(images) * train_ratio)
            
            train_images = images[:split_index]
            test_images = images[split_index:]
            
            os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)
            
            for img in train_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_folder, img))
            for img in test_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_folder, img))

source_directory = r'D:\dataset\all\FatsiaSprouts'
train_directory = r'D:\dataset\all\new_split\train'
test_directory = r'D:\dataset\all\new_split\test'

split_dataset(source_directory, train_directory, test_directory)
