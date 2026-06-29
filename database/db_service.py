import os
import json
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import streamlit as st

class DatabaseService:
    def __init__(self):
        # Try DATABASE_URL first (for Render deployment)
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            # Parse DATABASE_URL for Render deployment
            result = urlparse(database_url)
            self.connection_params = {
                'host': result.hostname,
                'port': result.port,
                'database': result.path[1:],  # Remove leading slash
                'user': result.username,
                'password': result.password
            }
        else:
            # Fall back to individual env vars (for Replit)
            self.connection_params = {
                'host': os.getenv('PGHOST'),
                'port': os.getenv('PGPORT'),
                'database': os.getenv('PGDATABASE'),
                'user': os.getenv('PGUSER'),
                'password': os.getenv('PGPASSWORD')
            }

        # Validate that we have required connection parameters
        if not database_url and not all([
            self.connection_params.get('host'),
            self.connection_params.get('database'),
            self.connection_params.get('user')
        ]):
            raise RuntimeError(
                "Database not configured. Please set DATABASE_URL environment variable "
                "or individual PostgreSQL connection variables (PGHOST, PGDATABASE, PGUSER, PGPASSWORD)."
            )

        # Make sure the schema exists. The pages use raw SQL (this service) which
        # assumes the tables are already present; nothing else creates them on a
        # fresh deployment, so do it here. UUID primary keys get a server-side
        # default because the raw INSERTs never supply an id.
        self._ensure_schema()

    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.connection_params)

    # SQL is split per statement so a failure on one CREATE doesn't abort the rest.
    _SCHEMA_STATEMENTS = [
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            total_images INTEGER,
            class_names TEXT[],
            class_distribution TEXT,
            created_at TIMESTAMP DEFAULT now(),
            updated_at TIMESTAMP DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS images (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            dataset_id UUID,
            filename VARCHAR(255),
            file_path VARCHAR(500),
            file_type VARCHAR(50),
            class_label VARCHAR(100),
            class_index INTEGER,
            width INTEGER,
            height INTEGER,
            channels INTEGER,
            file_size INTEGER,
            preprocessing_applied BOOLEAN DEFAULT FALSE,
            split_type VARCHAR(20),
            created_at TIMESTAMP DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS models (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            architecture VARCHAR(100),
            input_shape VARCHAR(50),
            num_classes INTEGER,
            total_parameters INTEGER,
            model_file_path VARCHAR(500),
            model_weights_path VARCHAR(500),
            is_trained BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT now(),
            updated_at TIMESTAMP DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS training_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id UUID,
            dataset_id UUID,
            epochs INTEGER,
            batch_size INTEGER,
            learning_rate DOUBLE PRECISION,
            dropout_rate DOUBLE PRECISION,
            l2_regularization DOUBLE PRECISION,
            use_data_augmentation BOOLEAN,
            use_class_weights BOOLEAN,
            class_weights TEXT,
            fine_tuning_enabled BOOLEAN,
            fine_tune_epochs INTEGER,
            fine_tune_learning_rate DOUBLE PRECISION,
            final_train_loss DOUBLE PRECISION,
            final_train_accuracy DOUBLE PRECISION,
            final_val_loss DOUBLE PRECISION,
            final_val_accuracy DOUBLE PRECISION,
            best_val_loss DOUBLE PRECISION,
            best_val_accuracy DOUBLE PRECISION,
            epochs_completed INTEGER,
            training_time_seconds DOUBLE PRECISION,
            training_history TEXT,
            status VARCHAR(50),
            error_message TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS model_evaluations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id UUID,
            evaluation_dataset VARCHAR(50),
            accuracy DOUBLE PRECISION,
            precision DOUBLE PRECISION,
            recall DOUBLE PRECISION,
            f1_score DOUBLE PRECISION,
            sensitivity DOUBLE PRECISION,
            specificity DOUBLE PRECISION,
            auc_roc DOUBLE PRECISION,
            auc_pr DOUBLE PRECISION,
            confusion_matrix TEXT,
            classification_report TEXT,
            per_class_metrics TEXT,
            total_samples INTEGER,
            evaluation_time_seconds DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id UUID,
            image_id UUID,
            external_filename VARCHAR(255),
            external_file_type VARCHAR(50),
            predicted_class VARCHAR(100),
            predicted_class_index INTEGER,
            confidence_score DOUBLE PRECISION,
            prediction_probabilities TEXT,
            prediction_time_seconds DOUBLE PRECISION,
            confidence_threshold DOUBLE PRECISION,
            preprocessing_applied BOOLEAN DEFAULT TRUE,
            batch_prediction BOOLEAN DEFAULT FALSE,
            batch_id UUID,
            risk_level VARCHAR(20),
            clinical_notes TEXT,
            created_at TIMESTAMP DEFAULT now()
        )
        """,
    ]

    def _ensure_schema(self):
        """Create tables if they don't exist. Failures are logged, not raised,
        so a schema hiccup never blocks the rest of the app from loading."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for statement in self._SCHEMA_STATEMENTS:
                        cur.execute(statement)
                conn.commit()
        except Exception as e:
            print(f"Schema initialization warning: {e}")
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        """Execute a query and return results"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params or ())
                    if fetch and cur.description:
                        columns = [desc[0] for desc in cur.description]
                        rows = cur.fetchall()
                        return pd.DataFrame(rows, columns=columns)
                    return None
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return None
    
    def save_dataset(self, name: str, X: np.ndarray, y: np.ndarray, class_names: List[str], description: str = None) -> str:
        """Save dataset to database"""
        try:
            class_distribution = {class_names[i]: int(np.sum(y == i)) for i in range(len(class_names))}
            
            query = """
            INSERT INTO datasets (name, description, total_images, class_names, class_distribution) 
            VALUES (%s, %s, %s, %s, %s) RETURNING id
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (
                        name,
                        description,
                        len(X),
                        class_names,
                        json.dumps(class_distribution)
                    ))
                    dataset_id = cur.fetchone()[0]
                    
                    # Save image metadata
                    for i, (image, label) in enumerate(zip(X, y)):
                        img_query = """
                        INSERT INTO images (dataset_id, filename, file_type, class_label, class_index, 
                                          width, height, channels, preprocessing_applied)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        cur.execute(img_query, (
                            dataset_id,
                            f"image_{i:06d}",
                            "processed",
                            class_names[int(label)],
                            int(label),
                            int(image.shape[1]),
                            int(image.shape[0]),
                            int(image.shape[2]) if len(image.shape) > 2 else 1,
                            True
                        ))
                    
                    conn.commit()
                    return str(dataset_id)
                    
        except Exception as e:
            st.error(f"Error saving dataset: {str(e)}")
            return None
    
    def save_model(self, name: str, architecture: str, input_shape: tuple, num_classes: int) -> str:
        """Save model metadata"""
        try:
            query = """
            INSERT INTO models (name, architecture, input_shape, num_classes) 
            VALUES (%s, %s, %s, %s) RETURNING id
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (name, architecture, str(input_shape), num_classes))
                    model_id = cur.fetchone()[0]
                    conn.commit()
                    return str(model_id)
                    
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            return None
    
    def save_training_session(self, model_id: str, dataset_id: str, params: Dict, results: Dict = None) -> str:
        """Save training session"""
        try:
            query = """
            INSERT INTO training_sessions (
                model_id, dataset_id, epochs, batch_size, learning_rate, dropout_rate,
                l2_regularization, use_data_augmentation, use_class_weights, class_weights,
                fine_tuning_enabled, fine_tune_epochs, fine_tune_learning_rate,
                final_train_loss, final_train_accuracy, final_val_loss, final_val_accuracy,
                training_history, status, started_at, completed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (
                        model_id, dataset_id,
                        params.get('epochs'), params.get('batch_size'), params.get('learning_rate'),
                        params.get('dropout_rate'), params.get('l2_reg'),
                        params.get('use_augmentation', False), params.get('use_class_weights', False),
                        json.dumps(params.get('class_weights', {})),
                        params.get('fine_tune', False), params.get('fine_tune_epochs'),
                        params.get('fine_tune_lr'),
                        results.get('final_train_loss') if results else None,
                        results.get('final_train_accuracy') if results else None,
                        results.get('final_val_loss') if results else None,
                        results.get('final_val_accuracy') if results else None,
                        json.dumps(results.get('history', {})) if results else None,
                        'completed' if results else 'initialized',
                        datetime.utcnow(),
                        datetime.utcnow() if results else None
                    ))
                    session_id = cur.fetchone()[0]
                    conn.commit()
                    return str(session_id)
                    
        except Exception as e:
            st.error(f"Error saving training session: {str(e)}")
            return None
    
    def save_evaluation(self, model_id: str, results: Dict) -> str:
        """Save model evaluation results"""
        try:
            query = """
            INSERT INTO model_evaluations (
                model_id, evaluation_dataset, accuracy, precision, recall, f1_score,
                sensitivity, specificity, auc_roc, auc_pr, confusion_matrix, 
                classification_report, total_samples
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (
                        model_id, 'test',
                        results.get('accuracy'), results.get('precision'), results.get('recall'),
                        results.get('f1_score'), results.get('sensitivity'), results.get('specificity'),
                        results.get('auc'), results.get('pr_auc'),
                        json.dumps(results.get('confusion_matrix', []).tolist() if hasattr(results.get('confusion_matrix', []), 'tolist') else []),
                        json.dumps(results.get('classification_report', {})),
                        len(results.get('y_true', []))
                    ))
                    eval_id = cur.fetchone()[0]
                    conn.commit()
                    return str(eval_id)
                    
        except Exception as e:
            st.error(f"Error saving evaluation: {str(e)}")
            return None
    
    def save_prediction(self, model_id: str, prediction_data: Dict) -> str:
        """Save prediction result"""
        try:
            # Calculate risk level
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
            
            query = """
            INSERT INTO predictions (
                model_id, external_filename, external_file_type, predicted_class,
                predicted_class_index, confidence_score, prediction_probabilities,
                confidence_threshold, risk_level
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (
                        model_id,
                        prediction_data.get('filename'),
                        prediction_data.get('file_type'),
                        prediction_data.get('predicted_class_name'),
                        predicted_class,
                        confidence,
                        json.dumps(prediction_data.get('probabilities', []).tolist() 
                                 if hasattr(prediction_data.get('probabilities', []), 'tolist') 
                                 else prediction_data.get('probabilities', [])),
                        prediction_data.get('confidence_threshold', 0.5),
                        risk_level
                    ))
                    pred_id = cur.fetchone()[0]
                    conn.commit()
                    return str(pred_id)
                    
        except Exception as e:
            st.error(f"Error saving prediction: {str(e)}")
            return None
    
    def get_datasets(self) -> pd.DataFrame:
        """Get all datasets"""
        query = "SELECT * FROM datasets ORDER BY created_at DESC"
        return self.execute_query(query)
    
    def get_models(self) -> pd.DataFrame:
        """Get all models"""
        query = "SELECT * FROM models ORDER BY created_at DESC"
        return self.execute_query(query)
    
    def get_training_sessions(self, model_id: str = None) -> pd.DataFrame:
        """Get training sessions"""
        if model_id:
            query = "SELECT * FROM training_sessions WHERE model_id = %s ORDER BY created_at DESC"
            return self.execute_query(query, (model_id,))
        else:
            query = "SELECT * FROM training_sessions ORDER BY created_at DESC"
            return self.execute_query(query)
    
    def get_predictions(self, model_id: str = None, limit: int = 100) -> pd.DataFrame:
        """Get predictions"""
        if model_id:
            query = "SELECT * FROM predictions WHERE model_id = %s ORDER BY created_at DESC LIMIT %s"
            return self.execute_query(query, (model_id, limit))
        else:
            query = "SELECT * FROM predictions ORDER BY created_at DESC LIMIT %s"
            return self.execute_query(query, (limit,))
    
    def get_dashboard_stats(self) -> Dict:
        """Get dashboard statistics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get various counts
                    cur.execute("SELECT COUNT(*) FROM datasets")
                    total_datasets = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM models")
                    total_models = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM models WHERE is_trained = TRUE")
                    trained_models = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM predictions")
                    total_predictions = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM predictions WHERE DATE(created_at) = CURRENT_DATE")
                    today_predictions = cur.fetchone()[0]
                    
                    return {
                        'total_datasets': total_datasets,
                        'total_models': total_models,
                        'trained_models': trained_models,
                        'total_predictions': total_predictions,
                        'today_predictions': today_predictions
                    }
        except Exception as e:
            st.error(f"Error getting dashboard stats: {str(e)}")
            return {}
    
    def update_model_status(self, model_id: str, is_trained: bool = True):
        """Update model training status"""
        try:
            query = "UPDATE models SET is_trained = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (is_trained, model_id))
                    conn.commit()
        except Exception as e:
            st.error(f"Error updating model status: {str(e)}")