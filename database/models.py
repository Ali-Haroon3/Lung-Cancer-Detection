import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    total_images = Column(Integer, nullable=False)
    class_names = Column(ARRAY(String), nullable=False)
    class_distribution = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    images = relationship("Image", back_populates="dataset")
    training_sessions = relationship("TrainingSession", back_populates="dataset")

class Image(Base):
    __tablename__ = "images"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500))
    file_type = Column(String(50))  # DICOM, PNG, JPG, etc.
    class_label = Column(String(100), nullable=False)
    class_index = Column(Integer, nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    channels = Column(Integer)
    file_size = Column(Integer)  # in bytes
    preprocessing_applied = Column(Boolean, default=False)
    split_type = Column(String(20))  # train, validation, test
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="images")
    predictions = relationship("Prediction", back_populates="image")

class Model(Base):
    __tablename__ = "models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    architecture = Column(String(100), nullable=False)  # resnet50, densenet121, etc.
    input_shape = Column(String(50))  # (224, 224, 3)
    num_classes = Column(Integer, nullable=False)
    total_parameters = Column(Integer)
    model_file_path = Column(String(500))
    model_weights_path = Column(String(500))
    is_trained = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    training_sessions = relationship("TrainingSession", back_populates="model")
    evaluations = relationship("ModelEvaluation", back_populates="model")
    predictions = relationship("Prediction", back_populates="model")

class TrainingSession(Base):
    __tablename__ = "training_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    
    # Training parameters
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    dropout_rate = Column(Float)
    l2_regularization = Column(Float)
    use_data_augmentation = Column(Boolean, default=False)
    use_class_weights = Column(Boolean, default=False)
    class_weights = Column(Text)  # JSON string
    
    # Fine-tuning parameters
    fine_tuning_enabled = Column(Boolean, default=False)
    fine_tune_epochs = Column(Integer)
    fine_tune_learning_rate = Column(Float)
    
    # Training results
    final_train_loss = Column(Float)
    final_train_accuracy = Column(Float)
    final_val_loss = Column(Float)
    final_val_accuracy = Column(Float)
    best_val_loss = Column(Float)
    best_val_accuracy = Column(Float)
    epochs_completed = Column(Integer)
    training_time_seconds = Column(Float)
    
    # Training history (JSON string)
    training_history = Column(Text)
    
    # Status and metadata
    status = Column(String(50), default="initialized")  # initialized, running, completed, failed
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("Model", back_populates="training_sessions")
    dataset = relationship("Dataset", back_populates="training_sessions")

class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    evaluation_dataset = Column(String(50))  # test, validation
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Medical-specific metrics
    sensitivity = Column(Float)  # Same as recall for binary classification
    specificity = Column(Float)
    auc_roc = Column(Float)
    auc_pr = Column(Float)  # Area under Precision-Recall curve
    
    # Detailed results (JSON strings)
    confusion_matrix = Column(Text)
    classification_report = Column(Text)
    per_class_metrics = Column(Text)
    
    # Metadata
    total_samples = Column(Integer)
    evaluation_time_seconds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("Model", back_populates="evaluations")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=True)  # Null for external images
    
    # Image information (for external images not in dataset)
    external_filename = Column(String(255))
    external_file_type = Column(String(50))
    
    # Prediction results
    predicted_class = Column(String(100), nullable=False)
    predicted_class_index = Column(Integer, nullable=False)
    confidence_score = Column(Float, nullable=False)
    prediction_probabilities = Column(Text)  # JSON array of probabilities
    
    # Additional metadata
    prediction_time_seconds = Column(Float)
    confidence_threshold = Column(Float)
    preprocessing_applied = Column(Boolean, default=True)
    batch_prediction = Column(Boolean, default=False)
    batch_id = Column(UUID(as_uuid=True))  # For grouping batch predictions
    
    # Risk assessment (for medical context)
    risk_level = Column(String(20))  # low, medium, high
    clinical_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("Model", back_populates="predictions")
    image = relationship("Image", back_populates="predictions")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_session = Column(String(255))  # Streamlit session ID
    action = Column(String(100), nullable=False)  # upload_data, train_model, make_prediction, etc.
    entity_type = Column(String(50))  # dataset, model, prediction
    entity_id = Column(UUID(as_uuid=True))
    details = Column(Text)  # JSON string with additional details
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create all tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Database session management
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session():
    return SessionLocal()