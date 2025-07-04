# Streamlit Community Cloud Deployment Guide

## Quick Deployment Steps

### Step 1: Deploy to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select "From existing repo"
5. Choose repository: `Ali-Haroon3/Lung-Cancer-Detection`
6. Branch: `main`
7. Main file path: `app.py`
8. Click "Deploy!"

### Step 2: Configure Environment Variables
Once deployed, go to app settings and add:
```
DATABASE_URL=your_postgresql_connection_string
```

### Step 3: Test Your Application
Your app will be live at: `https://[app-name].streamlit.app`
Test all features:
- Data upload and preprocessing
- Model training
- Predictions
- Database connectivity

### Step 4: Custom Domain Setup
1. **Purchase Domain**: Buy from Namecheap, GoDaddy, or similar
   - Suggested domains: `lungcancer-detector.com`, `medicalai-diagnostics.com`

2. **Configure in Streamlit Cloud**:
   - Go to app settings → "Custom Domain"
   - Add your domain: `yourdomain.com`
   - Streamlit provides DNS records

3. **Update DNS at Registrar**:
   Add these records at your domain provider:
   ```
   Type: CNAME
   Name: @
   Value: [provided by Streamlit Cloud]
   
   Type: CNAME
   Name: www
   Value: [provided by Streamlit Cloud]
   ```

4. **SSL Certificate**: Automatically provided by Streamlit Cloud

## Repository Requirements ✅

Your repository already has everything needed:
- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - All dependencies listed
- ✅ `.streamlit/config.toml` - Server configuration
- ✅ Database models and preprocessing utilities
- ✅ Pre-trained model files (via Git LFS)

## Expected Timeline
- **Deployment**: 2-3 minutes
- **Domain configuration**: 5-10 minutes
- **DNS propagation**: 10-60 minutes
- **Total time to live**: ~1 hour maximum

## Troubleshooting

### Build Errors
- Check `requirements.txt` for any missing dependencies
- Verify Python version compatibility
- Review build logs in Streamlit Cloud dashboard

### Database Connection
- Ensure DATABASE_URL is correctly set in app settings
- Verify PostgreSQL database is accessible from external connections
- Check connection string format

### Model Loading
- Git LFS files should load automatically
- If model files missing, check LFS configuration
- Verify `best_model.h5` is in repository

## Cost Breakdown
- **Streamlit Community Cloud**: FREE
- **Custom Domain**: $10-15/year for .com
- **Total Annual Cost**: $10-15/year

## Professional Domain Suggestions
- `lungcancer-detector.com`
- `medicalai-diagnostics.com` 
- `thoracic-imaging-ai.com`
- `pulmonary-detection-ai.com`
- `clinical-ai-tools.com`

Your lung cancer detection AI will be fully functional with real-time training, database integration, and professional custom domain!