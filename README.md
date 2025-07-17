# 🫁 Lung Cancer Detection AI

An advanced deep learning application for lung cancer detection using Convolutional Neural Networks (CNN) with medical imaging analysis.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.46+-red.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.17+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Live Demo

**[Try the Application](https://your-app-name.onrender.com)** *(Deploy to get your link)*

## 📋 Features

### 🔬 Medical AI Capabilities
- **Deep Learning Models**: ResNet50, DenseNet121, EfficientNetB0 with transfer learning
- **Medical Image Processing**: DICOM files, CT scans, chest X-rays
- **Real-time Predictions**: Single image and batch processing
- **Class Activation Maps**: Visual interpretability for medical decisions

### 💾 Data Management
- **PostgreSQL Integration**: Complete medical workflow tracking
- **Training History**: Full model performance analytics
- **Audit Trail**: Comprehensive logging for medical compliance
- **Data Export**: Results export for clinical workflows

### 🖥️ User Interface
- **Interactive Web App**: Streamlit-based medical interface
- **Multi-page Design**: Organized workflow for medical professionals
- **Real-time Visualization**: Training progress and results
- **Mobile Responsive**: Access from any device

## 🏥 Medical Use Case

This application assists medical professionals in:
- **Screening**: Early detection support for lung cancer
- **Education**: Understanding AI diagnostic tools
- **Research**: Model training with medical datasets
- **Workflow Integration**: Clinical decision support

*⚠️ Note: This is a research/educational tool. Always consult medical professionals for actual diagnosis.*

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **AI/ML**: TensorFlow, Keras, scikit-learn
- **Medical Imaging**: OpenCV, PIL, pydicom
- **Database**: PostgreSQL with SQLAlchemy
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📥 Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL (optional, for data persistence)
- 4GB+ RAM (recommended for model training)

### Local Development
```bash
# Clone the repository
git clone https://github.com/Ali-Haroon3/lung-cancer-detection.git
cd lung-cancer-detection

# Install dependencies
pip install -r requirements_streamlit.txt

# Set up environment variables (optional)
export DATABASE_URL="postgresql://user:password@localhost:5432/lung_cancer_db"

# Run the application
streamlit run app.py
```

### 🌐 Deploy to Render (Recommended)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Fork this repository
2. Go to [render.com](https://render.com)
3. Create new "Web Service" from your fork
4. Use build command: `pip install -r requirements_streamlit.txt`
5. Use start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

## 📖 Usage Guide

### 1. Data Upload
- Upload medical images (PNG, JPG, DICOM)
- Support for ZIP archives and batch processing
- Automatic preprocessing and validation

### 2. Model Training
- Select from pre-trained architectures
- Configure hyperparameters
- Real-time training progress monitoring
- Automatic model checkpointing

### 3. Evaluation & Analysis
- Medical performance metrics (sensitivity, specificity)
- ROC curves and confusion matrices
- Model comparison tools

### 4. Prediction & Diagnosis
- Single image and batch predictions
- Confidence scoring with medical thresholds
- Visual interpretability features
- Results export for clinical use

### 5. Database Management
- Training session tracking
- Performance analytics
- Data visualization dashboard

## 🏗️ Project Structure

```
lung-cancer-detection/
├── app.py                 # Main application entry point
├── pages/                 # Streamlit multi-page structure
│   ├── 1_Data_Upload.py
│   ├── 2_Model_Training.py
│   ├── 3_Model_Evaluation.py
│   ├── 4_Prediction.py
│   └── 5_Database_Management.py
├── models/                # CNN model implementations
│   └── cnn_model.py
├── database/              # Database service layer
│   ├── database_manager.py
│   ├── db_service.py
│   └── models.py
├── utils/                 # Utility modules
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   └── visualization.py
├── sample_data/           # Sample medical imaging data
└── .streamlit/           # Streamlit configuration
    └── config.toml
```

## 🔧 Configuration

### Environment Variables
```bash
DATABASE_URL=postgresql://user:password@host:port/database  # Optional
PYTHONPATH=.                                                # For imports
```

### Streamlit Configuration
The app includes optimized configuration for production deployment in `.streamlit/config.toml`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

This application is for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions regarding medical conditions.

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Ali-Haroon3/lung-cancer-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ali-Haroon3/lung-cancer-detection/discussions)

---

**Built with ❤️ for advancing medical AI research**