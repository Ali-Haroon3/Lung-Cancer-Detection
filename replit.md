# Lung Cancer Detection CNN Application

## Overview

This is a Streamlit-based web application that implements deep learning models for lung cancer detection from medical imaging data. The application uses Convolutional Neural Networks (CNN) with transfer learning to analyze CT scans and chest X-rays, providing medical professionals with AI-assisted diagnostic capabilities.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework providing an interactive medical imaging interface
- **Multi-page Structure**: Organized into distinct pages for data upload, model training, evaluation, prediction, and database management
- **Real-time Visualization**: Interactive charts and medical image displays using matplotlib, seaborn, and plotly
- **Session State Management**: Persistent data storage across page navigation for model state and processed data

### Backend Architecture
- **Deep Learning Framework**: TensorFlow/Keras for CNN implementation and model training
- **Transfer Learning**: Pre-trained models (ResNet50, DenseNet121, EfficientNetB0) fine-tuned for medical imaging
- **Medical Image Processing**: Specialized preprocessing pipeline for DICOM files and standard image formats
- **Model Architecture**: Custom CNN wrapper class supporting multiple architectures with medical-specific configurations

### Database Architecture
- **Database Engine**: PostgreSQL with complete relational schema for medical AI workflows
- **Core Tables**: datasets, models, images, training_sessions, model_evaluations, predictions, audit_logs
- **Data Persistence**: Full tracking of datasets, model metadata, training history, and prediction results
- **Analytics Support**: Built-in database analytics with interactive dashboards and performance metrics
- **Audit Trail**: Comprehensive logging of all user actions and system operations

### Data Processing Pipeline
- **Input Formats**: Support for DICOM, PNG, JPG medical images
- **Preprocessing**: Standardized image resizing, normalization, and medical-specific augmentation
- **Data Validation**: Medical imaging quality checks and format validation
- **Train/Validation/Test Split**: Stratified splitting with class balancing for medical datasets
- **Database Integration**: Automatic storage of processed datasets with metadata and class distributions

## Key Components

### 1. Main Application (`app.py`)
- Entry point with application configuration and overview
- Session state initialization for model persistence
- Medical application branding and feature documentation

### 2. CNN Model Implementation (`models/cnn_model.py`)
- `LungCancerCNN` class with configurable architectures
- Transfer learning implementation with medical-specific fine-tuning
- Regularization techniques (dropout, L2) suitable for medical data
- Support for binary and multi-class classification

### 3. Data Upload and Preprocessing (`pages/1_Data_Upload.py`)
- Medical image upload interface (individual files, ZIP archives, sample datasets)
- DICOM file support with metadata extraction
- Real-time preprocessing and data validation
- Dataset statistics and class distribution visualization

### 4. Model Training (`pages/2_Model_Training.py`)
- Interactive training configuration (architecture selection, hyperparameters)
- Real-time training progress monitoring
- Class weighting for imbalanced medical datasets
- Training history visualization and model checkpointing

### 5. Model Evaluation (`pages/3_Model_Evaluation.py`)
- Medical-specific evaluation metrics (sensitivity, specificity, AUC)
- Confusion matrix and ROC curve analysis
- Performance comparison tools
- Model interpretability features

### 6. Prediction Interface (`pages/4_Prediction.py`)
- Single image and batch prediction capabilities
- Confidence score display with medical decision thresholds
- Prediction result visualization and export
- Integration with clinical workflow considerations
- Automatic database storage of prediction results with risk assessment

### 7. Database Management (`pages/5_Database_Management.py`)
- Interactive database dashboard with real-time statistics
- Data visualization for datasets, models, training sessions, and predictions
- Advanced analytics including activity timelines and performance metrics
- Database health monitoring and management tools
- Export capabilities for analysis and reporting

### 8. Database Service Layer (`database/db_service.py`)
- PostgreSQL integration with comprehensive data persistence
- RESTful-style operations for all medical AI workflow components
- Transaction management and error handling
- Performance optimization with proper indexing
- Data integrity enforcement and validation

### 9. Utility Modules
- **Data Preprocessing** (`utils/data_preprocessing.py`): Medical image processing, DICOM handling, augmentation
- **Evaluation** (`utils/evaluation.py`): Medical performance metrics, statistical analysis
- **Visualization** (`utils/visualization.py`): Medical imaging displays, result visualization

## Data Flow

1. **Data Ingestion**: Medical images uploaded through Streamlit interface
2. **Preprocessing**: DICOM parsing, image standardization, quality validation
3. **Dataset Preparation**: Train/validation/test splitting with medical considerations
4. **Model Training**: Transfer learning with medical-specific fine-tuning
5. **Evaluation**: Medical performance metrics calculation and visualization
6. **Prediction**: Real-time inference with confidence scoring for clinical use

## External Dependencies

### Core ML/AI Libraries
- **TensorFlow/Keras**: Deep learning framework and model implementation
- **scikit-learn**: Data preprocessing, evaluation metrics, and utilities
- **OpenCV**: Image processing and computer vision operations
- **PIL/Pillow**: Image manipulation and format handling

### Medical Imaging
- **pydicom**: DICOM file reading and metadata extraction
- **numpy**: Numerical computing for image array operations

### Web Framework and Visualization
- **Streamlit**: Web application framework and UI components
- **matplotlib/seaborn**: Statistical visualization and medical charts
- **plotly**: Interactive visualization for medical data analysis

### Data Science
- **pandas**: Data manipulation and analysis
- **scipy**: Scientific computing for statistical operations

## Deployment Strategy

### Development Environment
- **Platform**: Replit-based development with streamlit run app.py
- **Dependencies**: Requirements managed through pip/conda
- **Resource Management**: GPU support for model training when available

### Production Considerations
- **Hosting Platform**: Configured for Render.com deployment (recommended)
- **Alternative Platforms**: Streamlit Community Cloud, Railway, Hugging Face Spaces
- **Database**: PostgreSQL integration with automatic environment variable setup
- **Performance**: Optimized for cloud deployment with proper resource management
- **Security**: Medical data handling compliance and privacy protection
- **Scalability**: Auto-scaling capabilities through cloud hosting platforms

### Model Persistence
- **Session State**: In-memory model storage during application session
- **Model Checkpointing**: Automatic saving during training process
- **Export Capabilities**: Model and prediction result export functionality

## Changelog

Recent Changes:
- July 17, 2025: GitHub deployment setup completed
  - Fixed streamlit command execution by using python -m streamlit 
  - Updated port configuration from 8501 to 5000 for proper deployment
  - Created comprehensive deployment files for Render hosting
  - Added render.yaml for automated deployment configuration
  - Created professional README_GITHUB.md for repository presentation
  - Added .gitignore to exclude Replit-specific files
  - Set up Procfile for various hosting platforms
  - Created detailed GITHUB_SETUP.md with step-by-step instructions
  - Configured app for production hosting with proper environment handling
  - Ready for deployment on Render, Streamlit Cloud, or Railway platforms
- June 30, 2025: GitHub integration setup completed
  - Created comprehensive README.md with project overview and technical documentation
  - Added .gitignore to exclude replit.md and protect sensitive files
  - Configured GITHUB_TOKEN for authentication
  - Created troubleshooting guides for Git integration issues
  - Replit Git service experiencing timeout issues - manual download/sync recommended
  - Repository ready for sync with complete project structure
- June 30, 2025: Major training and evaluation fixes completed
  - Fixed infinite training restart loop by removing problematic st.rerun() call
  - Training now progresses correctly through epochs without getting stuck
  - Fixed session state consistency across all pages (standardized to 'trained_model')
  - Model Evaluation page now recognizes trained models properly
  - Added sample image selector to Prediction page for easy testing
  - Training successfully completed with early stopping and model checkpointing
  - Model achieved validation accuracy with ResNet50 transfer learning
  - All training data and results saved to PostgreSQL database
- June 29, 2025: Complete PostgreSQL database integration implemented
  - Created comprehensive database schema with 7 core tables
  - Added database service layer for all CRUD operations
  - Built interactive database management dashboard
  - Integrated database connectivity throughout application
  - Added real-time statistics and analytics visualization
- June 29, 2025: Initial application setup with CNN architecture

## User Preferences

Preferred communication style: Simple, everyday language.