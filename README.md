# Lung Cancer Detection CNN Application

A comprehensive Streamlit-based web application for lung cancer detection using deep learning and medical imaging analysis.

## 🏥 Overview

This application provides medical professionals with AI-assisted diagnostic capabilities for lung cancer detection from CT scans and chest X-rays. Built with modern deep learning frameworks and featuring a complete medical workflow management system.

## ✨ Key Features

### Deep Learning Models
- **CNN Architectures**: ResNet50, DenseNet121, EfficientNetB0
- **Transfer Learning**: Pre-trained models fine-tuned for medical imaging
- **Real-time Training**: Live progress monitoring with early stopping
- **Model Persistence**: Automatic saving and loading of trained models

### Medical Image Processing
- **DICOM Support**: Native DICOM file reading and metadata extraction
- **Image Preprocessing**: Standardized resizing, normalization, and augmentation
- **Data Validation**: Medical imaging quality checks and format validation
- **Batch Processing**: Support for multiple image analysis

### Comprehensive Interface
- **Data Upload**: Individual files, ZIP archives, and sample datasets
- **Model Training**: Interactive configuration with hyperparameter tuning
- **Model Evaluation**: Medical-specific metrics (sensitivity, specificity, AUC)
- **Prediction System**: Single image and batch prediction with confidence scoring
- **Database Dashboard**: Real-time analytics and performance monitoring

### Database Integration
- **PostgreSQL Backend**: Complete relational schema for medical workflows
- **Data Persistence**: Full tracking of datasets, models, training sessions, and results
- **Audit Trail**: Comprehensive logging of all user actions and system operations
- **Analytics**: Interactive visualizations and performance metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database
- 4GB+ RAM recommended for model training

### Installation
```bash
# Clone the repository
git clone https://github.com/aliharoon/lung-cancer-detection.git
cd lung-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Set up database
export DATABASE_URL="your_postgresql_connection_string"

# Run the application
streamlit run app.py
```

### Using Sample Data
The application includes synthetic lung cancer imaging data for demonstration:
```python
# Generate sample dataset
python sample_data_generator.py
```

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── pages/                    # Multi-page application structure
│   ├── 1_Data_Upload.py     # Data upload and preprocessing
│   ├── 2_Model_Training.py  # Model training interface
│   ├── 3_Model_Evaluation.py# Model evaluation and metrics
│   ├── 4_Prediction.py      # Prediction interface
│   └── 5_Database_Management.py # Database dashboard
├── models/                   # CNN model implementations
│   └── cnn_model.py         # LungCancerCNN class
├── database/                 # Database management
│   ├── models.py            # SQLAlchemy models
│   ├── database_manager.py  # Database operations
│   └── db_service.py        # Database service layer
├── utils/                    # Utility modules
│   ├── data_preprocessing.py# Medical image processing
│   ├── evaluation.py        # Performance metrics
│   └── visualization.py     # Medical visualization tools
├── sample_data/             # Sample medical images
└── requirements.txt         # Python dependencies
```

## 🔬 Technical Architecture

### Machine Learning Pipeline
1. **Data Ingestion**: Medical image upload with DICOM support
2. **Preprocessing**: Image standardization and quality validation
3. **Model Training**: Transfer learning with medical-specific fine-tuning
4. **Evaluation**: Comprehensive medical performance metrics
5. **Deployment**: Real-time inference with confidence scoring

### Database Schema
- **Datasets**: Image collections with metadata and class distributions
- **Models**: Model architecture and training configuration
- **Training Sessions**: Complete training history and hyperparameters
- **Evaluations**: Performance metrics and validation results
- **Predictions**: Inference results with confidence scores and risk assessment
- **Audit Logs**: User activity tracking and system monitoring

## 📊 Model Performance

### Supported Architectures
- **ResNet50**: Deep residual networks optimized for medical imaging
- **DenseNet121**: Dense connections for feature reuse and efficiency
- **EfficientNetB0**: Compound scaling for optimal accuracy/efficiency balance

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Medical-specific: Sensitivity, Specificity, AUC-ROC, AUC-PR
- Confusion matrices and ROC curve analysis
- Per-class performance breakdown

## 🛠️ Configuration

### Environment Variables
```bash
DATABASE_URL=postgresql://user:password@host:port/database
STREAMLIT_SERVER_PORT=5000
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Model Configuration
```python
# Training parameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
L2_REGULARIZATION = 0.001
```

## 🔒 Security & Compliance

- Database credentials secured through environment variables
- Medical data handling following privacy best practices
- Audit logging for regulatory compliance
- No patient data stored in version control

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or contributions:
- Create an issue in the GitHub repository
- Review the documentation in `/docs`
- Check the troubleshooting guide

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Streamlit team for the web application framework
- Medical imaging community for preprocessing best practices
- Open source contributors and medical AI researchers

---

**Note**: This application is for research and educational purposes. Medical decisions should always involve qualified healthcare professionals.