"""
Vercel API endpoint for Lung Cancer Detection AI Application
"""

def handler(request):
    """
    Main Vercel handler function
    """
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lung Cancer Detection AI - Medical Imaging Application</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                line-height: 1.6; 
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 1000px; 
                margin: 0 auto; 
                padding: 40px 20px;
            }
            .header {
                background: rgba(255, 255, 255, 0.95);
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                text-align: center;
                margin-bottom: 30px;
            }
            .title { 
                color: #2e7d32; 
                font-size: 3em; 
                margin-bottom: 10px;
                font-weight: 700;
            }
            .subtitle {
                color: #666;
                font-size: 1.3em;
                margin-bottom: 20px;
            }
            .content {
                background: rgba(255, 255, 255, 0.95);
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .feature {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 10px;
                border-left: 4px solid #007bff;
            }
            .feature h3 {
                color: #007bff;
                margin-bottom: 10px;
                font-size: 1.2em;
            }
            .deployment-notice {
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 20px;
                border-radius: 10px;
                margin: 30px 0;
            }
            .deployment-notice h3 {
                color: #856404;
                margin-bottom: 10px;
            }
            .links {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-top: 30px;
                flex-wrap: wrap;
            }
            .link {
                display: inline-block;
                padding: 15px 30px;
                background: linear-gradient(45deg, #007bff, #0056b3);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                font-weight: 600;
                transition: transform 0.3s ease;
            }
            .link:hover {
                transform: translateY(-3px);
                box-shadow: 0 5px 15px rgba(0,123,255,0.3);
            }
            .link.secondary {
                background: linear-gradient(45deg, #28a745, #20c997);
            }
            .tech-stack {
                margin-top: 30px;
                text-align: center;
            }
            .tech-stack h3 {
                margin-bottom: 15px;
                color: #333;
            }
            .tech-tags {
                display: flex;
                justify-content: center;
                gap: 10px;
                flex-wrap: wrap;
            }
            .tech-tag {
                background: #e9ecef;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                color: #495057;
            }
            @media (max-width: 768px) {
                .title { font-size: 2em; }
                .links { flex-direction: column; align-items: center; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 class="title">🏥 Lung Cancer Detection AI</h1>
                <p class="subtitle">Advanced Medical Imaging Analysis with Deep Learning</p>
            </div>
            
            <div class="content">
                <p style="font-size: 1.1em; text-align: center; margin-bottom: 30px;">
                    A sophisticated Streamlit-based medical imaging application that uses Convolutional Neural Networks 
                    to detect lung cancer from CT scans and chest X-rays, providing medical professionals with 
                    AI-assisted diagnostic capabilities.
                </p>
                
                <div class="features">
                    <div class="feature">
                        <h3>🧠 Advanced AI Models</h3>
                        <p>ResNet50-powered CNN with transfer learning for precise lung cancer classification and medical image analysis.</p>
                    </div>
                    <div class="feature">
                        <h3>🔬 Medical Image Processing</h3>
                        <p>Support for DICOM files, CT scans, and standard imaging formats with specialized preprocessing pipelines.</p>
                    </div>
                    <div class="feature">
                        <h3>📊 Real-time Training</h3>
                        <p>Interactive model training interface with live progress monitoring and performance visualization.</p>
                    </div>
                    <div class="feature">
                        <h3>💾 Database Integration</h3>
                        <p>PostgreSQL database for comprehensive tracking of datasets, models, training sessions, and predictions.</p>
                    </div>
                    <div class="feature">
                        <h3>📈 Evaluation Metrics</h3>
                        <p>Medical-specific metrics including sensitivity, specificity, AUC, and confusion matrix analysis.</p>
                    </div>
                    <div class="feature">
                        <h3>🎯 Clinical Predictions</h3>
                        <p>Confidence scoring and risk assessment for clinical decision support workflows.</p>
                    </div>
                </div>
                
                <div class="deployment-notice">
                    <h3>⚡ Deployment Information</h3>
                    <p>This Streamlit application requires specialized hosting configurations. For the full interactive experience, 
                    deploy on platforms optimized for Streamlit applications such as Streamlit Community Cloud, Heroku, or Railway.</p>
                </div>
                
                <div class="tech-stack">
                    <h3>Technology Stack</h3>
                    <div class="tech-tags">
                        <span class="tech-tag">Streamlit</span>
                        <span class="tech-tag">TensorFlow/Keras</span>
                        <span class="tech-tag">ResNet50</span>
                        <span class="tech-tag">PostgreSQL</span>
                        <span class="tech-tag">OpenCV</span>
                        <span class="tech-tag">DICOM</span>
                        <span class="tech-tag">scikit-learn</span>
                        <span class="tech-tag">Plotly</span>
                    </div>
                </div>
                
                <div class="links">
                    <a href="https://github.com/Ali-Haroon3/Lung-Cancer-Detection" class="link">📂 View Source Code</a>
                    <a href="https://streamlit.io/cloud" class="link secondary">🚀 Deploy on Streamlit Cloud</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html; charset=utf-8',
        },
        'body': html_content
    }