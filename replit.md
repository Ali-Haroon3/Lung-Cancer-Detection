# Lung Cancer Detection AI - System Architecture

## Overview

This is an advanced deep learning application for lung cancer detection using Convolutional Neural Networks (CNN) with medical imaging analysis. The system provides a complete workflow for medical professionals to upload medical images, train AI models, evaluate performance, and make predictions with comprehensive audit trails and database management.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit multi-page application
- **Structure**: Page-based navigation with 5 main sections:
  - Data Upload and Preprocessing
  - Model Training 
  - Model Evaluation
  - Prediction Interface
  - Database Management
- **UI Components**: Interactive forms, real-time visualizations, file upload widgets
- **State Management**: Streamlit session state for maintaining model and data across pages

### Backend Architecture
- **AI/ML Framework**: TensorFlow/Keras with transfer learning
- **Model Architectures**: ResNet50, DenseNet121, EfficientNetB0
- **Image Processing**: OpenCV, PIL for medical image preprocessing
- **Data Processing**: NumPy, pandas for data manipulation
- **Scientific Computing**: scikit-learn for evaluation metrics

### Database Architecture
- **Primary Database**: PostgreSQL
- **ORM**: SQLAlchemy with declarative models
- **Connection Management**: psycopg2-binary for direct PostgreSQL connections
- **Schema**: Comprehensive medical workflow tracking with entities for datasets, images, models, training sessions, evaluations, predictions, and audit logs

## Key Components

### 1. Data Management System
- **Medical Image Support**: DICOM, PNG, JPG, CT scans, chest X-rays
- **Upload Methods**: Individual files, ZIP archives, sample datasets
- **Preprocessing Pipeline**: Automated resizing, normalization, augmentation
- **Data Splitting**: Train/validation/test splits with stratification

### 2. AI Model Training System
- **Transfer Learning**: Pre-trained models from ImageNet
- **Model Architectures**: Multiple CNN options (ResNet50, DenseNet121, EfficientNetB0)
- **Training Features**: Early stopping, learning rate reduction, model checkpointing
- **Hyperparameter Control**: Dropout rates, L2 regularization, batch sizes

### 3. Model Evaluation System
- **Medical Metrics**: Comprehensive evaluation including sensitivity, specificity
- **Visualization Tools**: ROC curves, confusion matrices, classification reports
- **Performance Tracking**: Training history visualization and analysis
- **Clinical Interpretability**: Class activation maps for model decisions

### 4. Prediction System
- **Real-time Inference**: Single image and batch processing
- **Medical Interface**: User-friendly prediction interface for clinical use
- **Result Storage**: Comprehensive prediction logging and audit trails
- **Visualization**: Prediction confidence and interpretability features

### 5. Database Management System
- **Complete Workflow Tracking**: All medical AI operations logged
- **Analytics Dashboard**: Performance metrics and usage statistics
- **Data Export**: Clinical workflow integration capabilities
- **Audit Compliance**: Full audit trail for medical compliance requirements

## Data Flow

1. **Data Ingestion**: Medical images uploaded through Streamlit interface
2. **Preprocessing**: Images resized, normalized, and prepared for training
3. **Model Training**: CNN models trained with transfer learning
4. **Evaluation**: Model performance analyzed with medical-specific metrics
5. **Prediction**: Real-time inference on new medical images
6. **Storage**: All operations logged to PostgreSQL database
7. **Analytics**: Performance tracking and audit trail maintenance

## External Dependencies

### Core ML Libraries
- **TensorFlow 2.15.0**: Deep learning framework
- **scikit-learn 1.3.2**: Machine learning utilities and metrics
- **OpenCV**: Medical image processing
- **NumPy/pandas**: Data manipulation

### Medical Imaging
- **pydicom**: DICOM medical image format support
- **PIL/Pillow**: Image processing and manipulation

### Database
- **PostgreSQL**: Primary data storage
- **SQLAlchemy**: ORM and database abstraction
- **psycopg2-binary**: PostgreSQL adapter

### Visualization
- **Matplotlib/Seaborn**: Statistical plotting
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework

## Deployment Strategy

### Environment Configuration
- **Python 3.8+**: Runtime environment
- **Environment Variables**: Database connection parameters (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD)
- **Dependencies**: Managed through requirements_streamlit.txt for cloud deployment

### Database Setup
- **PostgreSQL Database**: Required for full functionality
- **Schema Creation**: Automated table creation on first run
- **Connection Management**: Robust error handling and connection pooling

### Scalability Considerations
- **Model Caching**: TensorFlow models cached in session state
- **Database Optimization**: Efficient queries and connection management
- **Image Processing**: Optimized preprocessing pipelines
- **Memory Management**: Careful handling of large medical image datasets

### Medical Compliance
- **Audit Trails**: Complete logging of all user actions
- **Data Privacy**: Secure handling of medical image data
- **Error Handling**: Robust error management for clinical environments
- **Performance Tracking**: Comprehensive model performance monitoring

The system is designed as a research and educational tool for medical professionals to understand and utilize AI diagnostic tools, with emphasis on clinical workflow integration and medical compliance requirements.