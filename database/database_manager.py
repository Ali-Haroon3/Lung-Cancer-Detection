import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from database.models import (
    Dataset, Image, Model, TrainingSession, ModelEvaluation, 
    Prediction, AuditLog, get_db_session, create_tables
)
import streamlit as st

class DatabaseManager:
    def __init__(self):
        """Initialize database manager and create tables if they don't exist"""
        try:
            create_tables()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def get_session(self) -> Session:
        """Get database session"""
        return get_db_session()
    
    def log_action(self, action: str, entity_type: str = None, entity_id: str = None, details: Dict = None):
        """Log user actions for audit trail"""
        try:
            db = self.get_session()
            log_entry = AuditLog(
                user_session=st.session_state.get('session_id', 'unknown'),
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                details=json.dumps(details) if details else None
            )
            db.add(log_entry)
            db.commit()
            db.close()
        except Exception as e:
            print(f"Logging error: {str(e)}")
    
    # Dataset Management
    def save_dataset(self, name: str, X: np.ndarray, y: np.ndarray, class_names: List[str], 
                    description: str = None) -> str:
        """Save dataset to database and return dataset ID"""
        db = self.get_session()
        try:
            # Create dataset record
            class_distribution = {class_names[i]: int(np.sum(y == i)) for i in range(len(class_names))}
            
            dataset = Dataset(
                name=name,
                description=description,
                total_images=len(X),
                class_names=class_names,
                class_distribution=json.dumps(class_distribution)
            )
            db.add(dataset)
            db.flush()  # Get the ID
            
            # Save individual images metadata
            for i, (image, label) in enumerate(zip(X, y)):
                image_record = Image(
                    dataset_id=dataset.id,
                    filename=f"image_{i:06d}",
                    file_type="processed",
                    class_label=class_names[label],
                    class_index=label,
                    width=image.shape[1],
                    height=image.shape[0],
                    channels=image.shape[2] if len(image.shape) > 2 else 1,
                    preprocessing_applied=True
                )
                db.add(image_record)
            
            db.commit()
            self.log_action("upload_dataset", "dataset", str(dataset.id), 
                          {"total_images": len(X), "classes": class_names})
            
            return str(dataset.id)
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_datasets(self) -> List[Dict]:
        """Get all datasets"""
        db = self.get_session()
        try:
            datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
            return [{
                'id': str(dataset.id),
                'name': dataset.name,
                'description': dataset.description,
                'total_images': dataset.total_images,
                'class_names': dataset.class_names,
                'class_distribution': json.loads(dataset.class_distribution) if dataset.class_distribution else {},
                'created_at': dataset.created_at
            } for dataset in datasets]
        finally:
            db.close()
    
    def update_image_splits(self, dataset_id: str, splits: Dict[str, np.ndarray]):
        """Update train/validation/test split information for images"""
        db = self.get_session()
        try:
            images = db.query(Image).filter(Image.dataset_id == dataset_id).all()
            
            # Create mapping of indices to split types
            split_mapping = {}
            offset = 0
            for split_type, indices in splits.items():
                for idx in indices:
                    split_mapping[offset + idx] = split_type
                if split_type == 'train':
                    offset = len(indices)
                elif split_type == 'val':
                    offset += len(indices)
            
            # Update images with split information
            for i, image in enumerate(images):
                if i in split_mapping:
                    image.split_type = split_mapping[i]
            
            db.commit()
        finally:
            db.close()
    
    # Model Management
    def save_model_metadata(self, name: str, architecture: str, input_shape: Tuple, 
                           num_classes: int, model_path: str = None) -> str:
        """Save model metadata and return model ID"""
        db = self.get_session()
        try:
            model = Model(
                name=name,
                architecture=architecture,
                input_shape=str(input_shape),
                num_classes=num_classes,
                model_file_path=model_path,
                is_trained=False
            )
            db.add(model)
            db.commit()
            
            self.log_action("create_model", "model", str(model.id), 
                          {"architecture": architecture, "input_shape": str(input_shape)})
            
            return str(model.id)
        finally:
            db.close()
    
    def update_model_training_status(self, model_id: str, is_trained: bool, model_path: str = None):
        """Update model training status"""
        db = self.get_session()
        try:
            model = db.query(Model).filter(Model.id == model_id).first()
            if model:
                model.is_trained = is_trained
                if model_path:
                    model.model_file_path = model_path
                model.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    
    def get_models(self) -> List[Dict]:
        """Get all models"""
        db = self.get_session()
        try:
            models = db.query(Model).order_by(Model.created_at.desc()).all()
            return [{
                'id': str(model.id),
                'name': model.name,
                'architecture': model.architecture,
                'input_shape': model.input_shape,
                'num_classes': model.num_classes,
                'is_trained': model.is_trained,
                'created_at': model.created_at,
                'updated_at': model.updated_at
            } for model in models]
        finally:
            db.close()
    
    # Training Session Management
    def save_training_session(self, model_id: str, dataset_id: str, training_params: Dict, 
                            training_results: Dict = None, training_history: Dict = None) -> str:
        """Save training session details"""
        db = self.get_session()
        try:
            session = TrainingSession(
                model_id=model_id,
                dataset_id=dataset_id,
                epochs=training_params.get('epochs'),
                batch_size=training_params.get('batch_size'),
                learning_rate=training_params.get('learning_rate'),
                dropout_rate=training_params.get('dropout_rate'),
                l2_regularization=training_params.get('l2_reg'),
                use_data_augmentation=training_params.get('use_augmentation', False),
                use_class_weights=training_params.get('use_class_weights', False),
                class_weights=json.dumps(training_params.get('class_weights', {})),
                fine_tuning_enabled=training_params.get('fine_tune', False),
                fine_tune_epochs=training_params.get('fine_tune_epochs'),
                fine_tune_learning_rate=training_params.get('fine_tune_lr'),
                status="initialized",
                started_at=datetime.utcnow()
            )
            
            if training_results:
                session.final_train_loss = training_results.get('final_train_loss')
                session.final_train_accuracy = training_results.get('final_train_accuracy')
                session.final_val_loss = training_results.get('final_val_loss')
                session.final_val_accuracy = training_results.get('final_val_accuracy')
                session.best_val_loss = training_results.get('best_val_loss')
                session.epochs_completed = training_results.get('epochs_completed')
                session.training_time_seconds = training_results.get('training_time')
                session.status = "completed"
                session.completed_at = datetime.utcnow()
            
            if training_history:
                session.training_history = json.dumps(training_history)
            
            db.add(session)
            db.commit()
            
            self.log_action("train_model", "training_session", str(session.id), training_params)
            
            return str(session.id)
        finally:
            db.close()
    
    def update_training_session_status(self, session_id: str, status: str, error_message: str = None):
        """Update training session status"""
        db = self.get_session()
        try:
            session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
            if session:
                session.status = status
                if error_message:
                    session.error_message = error_message
                if status == "completed":
                    session.completed_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    
    def get_training_sessions(self, model_id: str = None) -> List[Dict]:
        """Get training sessions, optionally filtered by model"""
        db = self.get_session()
        try:
            query = db.query(TrainingSession)
            if model_id:
                query = query.filter(TrainingSession.model_id == model_id)
            
            sessions = query.order_by(TrainingSession.created_at.desc()).all()
            return [{
                'id': str(session.id),
                'model_id': str(session.model_id),
                'dataset_id': str(session.dataset_id),
                'epochs': session.epochs,
                'batch_size': session.batch_size,
                'learning_rate': session.learning_rate,
                'final_val_accuracy': session.final_val_accuracy,
                'status': session.status,
                'started_at': session.started_at,
                'completed_at': session.completed_at,
                'training_time_seconds': session.training_time_seconds
            } for session in sessions]
        finally:
            db.close()
    
    # Model Evaluation Management
    def save_model_evaluation(self, model_id: str, evaluation_results: Dict, 
                            evaluation_dataset: str = "test") -> str:
        """Save model evaluation results"""
        db = self.get_session()
        try:
            evaluation = ModelEvaluation(
                model_id=model_id,
                evaluation_dataset=evaluation_dataset,
                accuracy=evaluation_results.get('accuracy'),
                precision=evaluation_results.get('precision'),
                recall=evaluation_results.get('recall'),
                f1_score=evaluation_results.get('f1_score'),
                sensitivity=evaluation_results.get('sensitivity'),
                specificity=evaluation_results.get('specificity'),
                auc_roc=evaluation_results.get('auc'),
                auc_pr=evaluation_results.get('pr_auc'),
                confusion_matrix=json.dumps(evaluation_results.get('confusion_matrix', []).tolist() 
                                          if 'confusion_matrix' in evaluation_results else None),
                classification_report=json.dumps(evaluation_results.get('classification_report', {})),
                total_samples=len(evaluation_results.get('y_true', [])),
                evaluation_time_seconds=evaluation_results.get('evaluation_time', 0)
            )
            
            db.add(evaluation)
            db.commit()
            
            self.log_action("evaluate_model", "model_evaluation", str(evaluation.id), 
                          {"model_id": model_id, "dataset": evaluation_dataset})
            
            return str(evaluation.id)
        finally:
            db.close()
    
    def get_model_evaluations(self, model_id: str) -> List[Dict]:
        """Get evaluations for a specific model"""
        db = self.get_session()
        try:
            evaluations = db.query(ModelEvaluation).filter(
                ModelEvaluation.model_id == model_id
            ).order_by(ModelEvaluation.created_at.desc()).all()
            
            return [{
                'id': str(eval.id),
                'evaluation_dataset': eval.evaluation_dataset,
                'accuracy': eval.accuracy,
                'precision': eval.precision,
                'recall': eval.recall,
                'f1_score': eval.f1_score,
                'sensitivity': eval.sensitivity,
                'specificity': eval.specificity,
                'auc_roc': eval.auc_roc,
                'auc_pr': eval.auc_pr,
                'total_samples': eval.total_samples,
                'created_at': eval.created_at
            } for eval in evaluations]
        finally:
            db.close()
    
    # Prediction Management
    def save_prediction(self, model_id: str, prediction_data: Dict, image_id: str = None) -> str:
        """Save prediction results"""
        db = self.get_session()
        try:
            # Calculate risk level for medical context
            confidence = prediction_data.get('confidence', 0)
            predicted_class = prediction_data.get('predicted_class', 0)
            
            if predicted_class == 1:  # Cancer predicted
                if confidence > 0.8:
                    risk_level = "high"
                elif confidence > 0.6:
                    risk_level = "medium"
                else:
                    risk_level = "low"
            else:
                risk_level = "low"
            
            prediction = Prediction(
                model_id=model_id,
                image_id=image_id,
                external_filename=prediction_data.get('filename'),
                external_file_type=prediction_data.get('file_type'),
                predicted_class=prediction_data.get('predicted_class_name'),
                predicted_class_index=prediction_data.get('predicted_class'),
                confidence_score=confidence,
                prediction_probabilities=json.dumps(prediction_data.get('probabilities', []).tolist() 
                                                   if hasattr(prediction_data.get('probabilities', []), 'tolist') 
                                                   else prediction_data.get('probabilities', [])),
                prediction_time_seconds=prediction_data.get('prediction_time', 0),
                confidence_threshold=prediction_data.get('confidence_threshold', 0.5),
                batch_prediction=prediction_data.get('batch_prediction', False),
                batch_id=prediction_data.get('batch_id'),
                risk_level=risk_level
            )
            
            db.add(prediction)
            db.commit()
            
            self.log_action("make_prediction", "prediction", str(prediction.id), 
                          {"model_id": model_id, "predicted_class": prediction_data.get('predicted_class_name')})
            
            return str(prediction.id)
        finally:
            db.close()
    
    def get_predictions(self, model_id: str = None, limit: int = 100) -> List[Dict]:
        """Get predictions, optionally filtered by model"""
        db = self.get_session()
        try:
            query = db.query(Prediction)
            if model_id:
                query = query.filter(Prediction.model_id == model_id)
            
            predictions = query.order_by(Prediction.created_at.desc()).limit(limit).all()
            
            return [{
                'id': str(pred.id),
                'model_id': str(pred.model_id),
                'filename': pred.external_filename,
                'predicted_class': pred.predicted_class,
                'confidence_score': pred.confidence_score,
                'risk_level': pred.risk_level,
                'created_at': pred.created_at,
                'batch_prediction': pred.batch_prediction
            } for pred in predictions]
        finally:
            db.close()
    
    # Statistics and Analytics
    def get_model_statistics(self, model_id: str) -> Dict:
        """Get comprehensive statistics for a model"""
        db = self.get_session()
        try:
            # Get model info
            model = db.query(Model).filter(Model.id == model_id).first()
            if not model:
                return {}
            
            # Get training sessions
            training_sessions = db.query(TrainingSession).filter(
                TrainingSession.model_id == model_id
            ).count()
            
            # Get evaluations
            evaluations = db.query(ModelEvaluation).filter(
                ModelEvaluation.model_id == model_id
            ).all()
            
            # Get predictions
            predictions = db.query(Prediction).filter(
                Prediction.model_id == model_id
            ).all()
            
            # Calculate prediction statistics
            total_predictions = len(predictions)
            cancer_predictions = sum(1 for p in predictions if p.predicted_class_index == 1)
            high_risk_predictions = sum(1 for p in predictions if p.risk_level == "high")
            
            return {
                'model_name': model.name,
                'architecture': model.architecture,
                'is_trained': model.is_trained,
                'training_sessions': training_sessions,
                'total_evaluations': len(evaluations),
                'best_accuracy': max([e.accuracy for e in evaluations], default=0),
                'best_auc': max([e.auc_roc for e in evaluations if e.auc_roc], default=0),
                'total_predictions': total_predictions,
                'cancer_predictions': cancer_predictions,
                'high_risk_predictions': high_risk_predictions,
                'created_at': model.created_at
            }
        finally:
            db.close()
    
    def get_dashboard_stats(self) -> Dict:
        """Get overall dashboard statistics"""
        db = self.get_session()
        try:
            total_datasets = db.query(Dataset).count()
            total_models = db.query(Model).count()
            trained_models = db.query(Model).filter(Model.is_trained == True).count()
            total_predictions = db.query(Prediction).count()
            recent_predictions = db.query(Prediction).filter(
                Prediction.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
            ).count()
            
            return {
                'total_datasets': total_datasets,
                'total_models': total_models,
                'trained_models': trained_models,
                'total_predictions': total_predictions,
                'recent_predictions': recent_predictions
            }
        finally:
            db.close()