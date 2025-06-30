import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFilter
import random

def create_sample_lung_images():
    """Create sample lung cancer imaging data for demonstration"""
    
    # Create directories
    os.makedirs("sample_data/lung_cancer_dataset/normal", exist_ok=True)
    os.makedirs("sample_data/lung_cancer_dataset/cancer", exist_ok=True)
    
    def create_lung_image(has_cancer=False, size=(224, 224)):
        """Create a synthetic lung image with or without cancer indicators"""
        # Create base lung structure
        img = np.zeros((*size, 3), dtype=np.uint8)
        
        # Add lung outline (circular/oval shape)
        center_x, center_y = size[0] // 2, size[1] // 2
        cv2.ellipse(img, (center_x, center_y), (80, 60), 0, 0, 360, (40, 40, 40), -1)
        
        # Add lung texture
        noise = np.random.normal(0, 15, size).astype(np.uint8)
        img = cv2.add(img, np.stack([noise, noise, noise], axis=2))
        
        # Add ribcage structure
        for i in range(5):
            y_pos = 30 + i * 30
            cv2.line(img, (20, y_pos), (size[0]-20, y_pos), (60, 60, 60), 1)
        
        if has_cancer:
            # Add cancer-like nodules
            num_nodules = random.randint(1, 3)
            for _ in range(num_nodules):
                x = random.randint(50, size[0] - 50)
                y = random.randint(50, size[1] - 50)
                radius = random.randint(8, 15)
                # Make nodules slightly brighter
                cv2.circle(img, (x, y), radius, (80, 80, 80), -1)
                # Add irregular edges
                cv2.circle(img, (x, y), radius + 2, (70, 70, 70), 1)
        
        # Add some random bright spots (normal tissue variations)
        for _ in range(random.randint(10, 20)):
            x = random.randint(0, size[0])
            y = random.randint(0, size[1])
            cv2.circle(img, (x, y), 1, (random.randint(80, 120),) * 3, -1)
        
        # Apply gaussian blur for realistic medical imaging effect
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
    # Generate normal lung images
    print("Creating normal lung images...")
    for i in range(50):
        img = create_lung_image(has_cancer=False)
        cv2.imwrite(f"sample_data/lung_cancer_dataset/normal/normal_{i:03d}.png", img)
    
    # Generate cancer lung images
    print("Creating cancer lung images...")
    for i in range(50):
        img = create_lung_image(has_cancer=True)
        cv2.imwrite(f"sample_data/lung_cancer_dataset/cancer/cancer_{i:03d}.png", img)
    
    print("Sample dataset created successfully!")
    print("Dataset structure:")
    print("- sample_data/lung_cancer_dataset/normal/ (50 images)")
    print("- sample_data/lung_cancer_dataset/cancer/ (50 images)")

if __name__ == "__main__":
    create_sample_lung_images()