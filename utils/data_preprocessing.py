import numpy as np
import cv2
import os
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    # Delay importing keras until needed to avoid deadlock
except ImportError:
    tf = None
    TF_AVAILABLE = False

class MedicalImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.label_encoder = LabelEncoder()
    
    def preprocess_images(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Preprocess images for medical analysis"""
        processed_images = []
        
        for image in X:
            # Resize image
            if len(image.shape) == 3:
                resized = cv2.resize(image, self.target_size)
            else:
                resized = cv2.resize(image, self.target_size)
                resized = np.expand_dims(resized, axis=-1)
                resized = np.repeat(resized, 3, axis=-1)
            
            # Normalize pixel values
            if normalize:
                resized = resized.astype(np.float32) / 255.0
            
            processed_images.append(resized)
        
        return np.array(processed_images)
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, 
                             batch_size=32, augment=True):
        """Create TensorFlow data generators"""
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available for data generators")
        
        # Import ImageDataGenerator only when needed to avoid import issues
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        if augment:
            train_datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()
        
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def prepare_data_splits(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def calculate_class_weights(self, y):
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced', classes=classes, y=y
        )
        
        return {i: class_weights[i] for i in range(len(classes))}
    
    def compute_class_weights(self, y):
        """Alias for calculate_class_weights for backward compatibility"""
        return self.calculate_class_weights(y)
    
    def split_dataset(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Alias for prepare_data_splits for backward compatibility"""
        return self.prepare_data_splits(X, y, test_size, val_size, random_state)
    
    def augment_minority_class(self, X, y, target_ratio=1.0):
        """Augment minority class to balance dataset"""
        unique, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        target_count = int(max_count * target_ratio)
        
        X_augmented = []
        y_augmented = []
        
        for class_label in unique:
            class_indices = np.where(y == class_label)[0]
            class_X = X[class_indices]
            class_y = y[class_indices]
            
            # Add original samples
            X_augmented.extend(class_X)
            y_augmented.extend(class_y)
            
            # Augment if needed
            current_count = len(class_X)
            if current_count < target_count:
                needed = target_count - current_count
                
                # Simple augmentation with rotation and flip
                for i in range(needed):
                    # Select random image from this class
                    idx = np.random.randint(0, len(class_X))
                    img = class_X[idx].copy()
                    
                    # Apply random transformation
                    if np.random.random() > 0.5:
                        img = np.fliplr(img)
                    
                    angle = np.random.randint(-15, 16)
                    if angle != 0:
                        h, w = img.shape[:2]
                        center = (w // 2, h // 2)
                        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img = cv2.warpAffine(img, matrix, (w, h))
                    
                    X_augmented.append(img)
                    y_augmented.append(class_label)
        
        return np.array(X_augmented), np.array(y_augmented)
    
    def create_dataset_from_directory(self, directory_path):
        """Create dataset from directory structure with class subdirectories"""
        X = []
        y = []
        class_names = []
        
        # Get class directories
        for class_name in sorted(os.listdir(directory_path)):
            class_path = os.path.join(directory_path, class_name)
            if os.path.isdir(class_path):
                class_names.append(class_name)
                class_index = len(class_names) - 1
                
                # Load images from class directory
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm')):
                        image_path = os.path.join(class_path, filename)
                        
                        try:
                            # Load image
                            if filename.lower().endswith('.dcm'):
                                # Handle DICOM files (would need pydicom)
                                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                                if image is not None:
                                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                            else:
                                image = cv2.imread(image_path)
                                if image is not None:
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            
                            if image is not None:
                                X.append(image)
                                y.append(class_index)
                        
                        except Exception as e:
                            print(f"Error loading {image_path}: {e}")
                            continue
        
        if len(X) == 0:
            raise ValueError("No valid images found in directory")
        
        # Convert to numpy arrays and preprocess
        X = np.array(X)
        y = np.array(y)
        
        # Preprocess images
        X = self.preprocess_images(X, normalize=True)
        
        return X, y, class_names
    
    def visualize_samples(self, X, y, class_names, num_samples=8):
        """Visualize sample images from the dataset"""
        import matplotlib.pyplot as plt
        
        # Select random samples for each class
        fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
        
        unique_classes = np.unique(y)
        samples_per_class = num_samples // len(unique_classes)
        
        sample_idx = 0
        for class_idx in unique_classes:
            class_indices = np.where(y == class_idx)[0]
            selected_indices = np.random.choice(class_indices, min(samples_per_class, len(class_indices)), replace=False)
            
            for i, idx in enumerate(selected_indices):
                if sample_idx >= num_samples:
                    break
                    
                row = sample_idx // (num_samples // 2)
                col = sample_idx % (num_samples // 2)
                
                # Display image
                img = X[idx]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                
                axes[row, col].imshow(img)
                axes[row, col].set_title(f'{class_names[class_idx]}')
                axes[row, col].axis('off')
                
                sample_idx += 1
        
        # Hide empty subplots
        for i in range(sample_idx, num_samples):
            row = i // (num_samples // 2)
            col = i % (num_samples // 2)
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    def load_standard_image(self, image_path):
        """Load a standard image file (PNG, JPG, etc.)"""
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_dicom(self, dicom_path):
        """Load a DICOM file"""
        try:
            import pydicom
            ds = pydicom.dcmread(dicom_path)
            image = ds.pixel_array
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            return image
        except ImportError:
            # Fallback to regular image loading if pydicom not available
            image = cv2.imread(dicom_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return image
    
    def load_and_preprocess(self, image_path, enhance_contrast=False):
        """Load and preprocess an image for prediction"""
        # Load image based on file extension
        if image_path.lower().endswith('.dcm'):
            image = self.load_dicom(image_path)
        else:
            image = self.load_standard_image(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize to target size
        image_resized = cv2.resize(image, self.target_size)
        
        # Enhance contrast if requested
        if enhance_contrast:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image_resized, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            image_resized = cv2.merge([l, a, b])
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_LAB2RGB)
        
        # Normalize to 0-1 range
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        return image_normalized