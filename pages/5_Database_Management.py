import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from database.db_service import DatabaseService

st.set_page_config(
    page_title="Database Management - Lung Cancer Detection",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ Database Management & Analytics")

# Initialize database service
@st.cache_resource
def get_db_service():
    try:
        return DatabaseService()
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

db_service = get_db_service()

if not db_service:
    st.error("Database service unavailable. Please check your database connection.")
    st.stop()

# Sidebar for database operations
st.sidebar.markdown("### Database Operations")
refresh_data = st.sidebar.button("🔄 Refresh Data")

# Main dashboard
st.markdown("### 📊 Database Overview")

try:
    # Get dashboard statistics
    stats = db_service.get_dashboard_stats()
    
    if stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Datasets", stats.get('total_datasets', 0))
        with col2:
            st.metric("Total Models", stats.get('total_models', 0))
        with col3:
            st.metric("Trained Models", stats.get('trained_models', 0))
        with col4:
            st.metric("Total Predictions", stats.get('total_predictions', 0))
        with col5:
            st.metric("Today's Predictions", stats.get('today_predictions', 0))
    
    # Create tabs for different data views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Datasets", "Models", "Training History", "Predictions", "Analytics"])
    
    with tab1:
        st.markdown("#### 📁 Datasets")
        
        try:
            datasets_df = db_service.get_datasets()
            
            if datasets_df is not None and not datasets_df.empty:
                # Display datasets table
                st.dataframe(
                    datasets_df[['name', 'total_images', 'class_names', 'created_at']],
                    use_container_width=True
                )
                
                # Dataset details
                if len(datasets_df) > 0:
                    st.markdown("#### Dataset Details")
                    selected_dataset = st.selectbox(
                        "Select dataset for details:",
                        options=range(len(datasets_df)),
                        format_func=lambda x: f"{datasets_df.iloc[x]['name']} ({datasets_df.iloc[x]['total_images']} images)"
                    )
                    
                    if selected_dataset is not None:
                        dataset_info = datasets_df.iloc[selected_dataset]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"""
                            **Dataset Name:** {dataset_info['name']}
                            **Total Images:** {dataset_info['total_images']}
                            **Classes:** {', '.join(dataset_info['class_names'])}
                            **Created:** {dataset_info['created_at']}
                            """)
                        
                        with col2:
                            if dataset_info['class_distribution']:
                                import json
                                try:
                                    class_dist = json.loads(dataset_info['class_distribution'])
                                    fig = px.pie(
                                        values=list(class_dist.values()),
                                        names=list(class_dist.keys()),
                                        title="Class Distribution"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except:
                                    st.warning("Could not display class distribution chart")
            else:
                st.info("No datasets found in the database.")
                
        except Exception as e:
            st.error(f"Error loading datasets: {str(e)}")
    
    with tab2:
        st.markdown("#### 🧠 Models")
        
        try:
            models_df = db_service.get_models()
            
            if models_df is not None and not models_df.empty:
                # Add status indicators
                models_display = models_df.copy()
                models_display['Status'] = models_display['is_trained'].apply(
                    lambda x: "✅ Trained" if x else "⚠️ Not Trained"
                )
                
                st.dataframe(
                    models_display[['name', 'architecture', 'num_classes', 'Status', 'created_at']],
                    use_container_width=True
                )
                
                # Model architecture distribution
                if len(models_df) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Architecture Distribution")
                        arch_counts = models_df['architecture'].value_counts()
                        fig = px.bar(
                            x=arch_counts.index,
                            y=arch_counts.values,
                            title="Models by Architecture"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Training Status")
                        status_counts = models_df['is_trained'].value_counts()
                        fig = px.pie(
                            values=status_counts.values,
                            names=['Trained' if x else 'Not Trained' for x in status_counts.index],
                            title="Model Training Status"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No models found in the database.")
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
    
    with tab3:
        st.markdown("#### 📈 Training History")
        
        try:
            training_df = db_service.get_training_sessions()
            
            if training_df is not None and not training_df.empty:
                # Display training sessions
                training_display = training_df.copy()
                training_display['Duration'] = training_display['training_time_seconds'].apply(
                    lambda x: f"{x:.1f}s" if pd.notna(x) else "N/A"
                )
                
                st.dataframe(
                    training_display[['model_id', 'epochs', 'batch_size', 'learning_rate', 
                                    'final_val_accuracy', 'status', 'Duration', 'started_at']],
                    use_container_width=True
                )
                
                # Training performance visualization
                if len(training_df) > 0:
                    # Filter completed sessions with accuracy data
                    completed_sessions = training_df[
                        (training_df['status'] == 'completed') & 
                        (training_df['final_val_accuracy'].notna())
                    ]
                    
                    if not completed_sessions.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Training Performance Over Time")
                            fig = px.scatter(
                                completed_sessions,
                                x='started_at',
                                y='final_val_accuracy',
                                title="Validation Accuracy Over Time",
                                hover_data=['epochs', 'learning_rate']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### Training Duration vs Accuracy")
                            fig = px.scatter(
                                completed_sessions,
                                x='training_time_seconds',
                                y='final_val_accuracy',
                                title="Training Time vs Accuracy",
                                hover_data=['epochs', 'batch_size']
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No training sessions found in the database.")
                
        except Exception as e:
            st.error(f"Error loading training history: {str(e)}")
    
    with tab4:
        st.markdown("#### 🔮 Predictions")
        
        try:
            predictions_df = db_service.get_predictions(limit=500)
            
            if predictions_df is not None and not predictions_df.empty:
                # Display recent predictions
                predictions_display = predictions_df.copy()
                predictions_display['Risk Level'] = predictions_display['risk_level'].apply(
                    lambda x: f"🔴 {x.upper()}" if x == 'high' else 
                              f"🟡 {x.upper()}" if x == 'medium' else f"🟢 {x.upper()}"
                )
                
                st.dataframe(
                    predictions_display[['external_filename', 'predicted_class', 'confidence_score', 
                                       'Risk Level', 'created_at']].head(50),
                    use_container_width=True
                )
                
                # Prediction analytics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Prediction Distribution")
                    pred_counts = predictions_df['predicted_class'].value_counts()
                    fig = px.pie(
                        values=pred_counts.values,
                        names=pred_counts.index,
                        title="Prediction Class Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Risk Level Distribution")
                    risk_counts = predictions_df['risk_level'].value_counts()
                    colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
                    fig = px.bar(
                        x=risk_counts.index,
                        y=risk_counts.values,
                        title="Risk Level Distribution",
                        color=risk_counts.index,
                        color_discrete_map=colors
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                st.markdown("#### Confidence Score Distribution")
                fig = px.histogram(
                    predictions_df,
                    x='confidence_score',
                    nbins=20,
                    title="Distribution of Prediction Confidence Scores"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No predictions found in the database.")
                
        except Exception as e:
            st.error(f"Error loading predictions: {str(e)}")
    
    with tab5:
        st.markdown("#### 📊 Advanced Analytics")
        
        # Time-based analytics
        st.markdown("#### Activity Timeline")
        
        try:
            # Get recent activity data
            recent_predictions = db_service.get_predictions(limit=100)
            
            if recent_predictions is not None and not recent_predictions.empty:
                # Convert timestamp and create daily counts
                recent_predictions['date'] = pd.to_datetime(recent_predictions['created_at']).dt.date
                daily_predictions = recent_predictions.groupby('date').size().reset_index(name='count')
                
                fig = px.line(
                    daily_predictions,
                    x='date',
                    y='count',
                    title="Daily Prediction Activity",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics over time
                st.markdown("#### System Performance Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_confidence = recent_predictions['confidence_score'].mean()
                    st.metric("Average Confidence", f"{avg_confidence:.3f}")
                
                with col2:
                    cancer_detection_rate = (recent_predictions['predicted_class'] == 'Cancer').mean() * 100
                    st.metric("Cancer Detection Rate", f"{cancer_detection_rate:.1f}%")
                
                with col3:
                    high_risk_rate = (recent_predictions['risk_level'] == 'high').mean() * 100
                    st.metric("High Risk Rate", f"{high_risk_rate:.1f}%")
            
        except Exception as e:
            st.error(f"Error generating analytics: {str(e)}")
        
        # Database health check
        st.markdown("---")
        st.markdown("#### 🔍 Database Health Check")
        
        if st.button("Run Health Check"):
            with st.spinner("Running database health check..."):
                try:
                    # Check table sizes
                    health_results = []
                    
                    # Get basic statistics
                    datasets_count = len(db_service.get_datasets()) if db_service.get_datasets() is not None else 0
                    models_count = len(db_service.get_models()) if db_service.get_models() is not None else 0
                    predictions_count = len(db_service.get_predictions(limit=10000)) if db_service.get_predictions(limit=10000) is not None else 0
                    
                    health_results.append({"Check": "Datasets Table", "Status": "✅ OK", "Count": datasets_count})
                    health_results.append({"Check": "Models Table", "Status": "✅ OK", "Count": models_count})
                    health_results.append({"Check": "Predictions Table", "Status": "✅ OK", "Count": predictions_count})
                    
                    health_df = pd.DataFrame(health_results)
                    st.dataframe(health_df, use_container_width=True)
                    
                    st.success("Database health check completed successfully!")
                    
                except Exception as e:
                    st.error(f"Health check failed: {str(e)}")

except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("app.py")
with col2:
    if st.button("Data Upload →", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")