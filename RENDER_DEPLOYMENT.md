# 🚀 Deploy Your Lung Cancer Detection AI on Render

## Why Render?
- ✅ **Completely FREE** (750 hours/month)
- ✅ **No credit card required**
- ✅ **Professional hosting** without branding
- ✅ **Auto-deploy from GitHub** 
- ✅ **Free PostgreSQL database**
- ✅ **Custom domains supported**

## Step-by-Step Deployment

### 1. Sign Up for Render
- Go to [render.com](https://render.com)
- Click **"Get Started for Free"**
- Sign up using your GitHub account (Ali-Haroon3)

### 2. Create Web Service
- Click **"New +"** in the dashboard
- Select **"Web Service"**
- Click **"Connect GitHub"** and authorize Render
- Select your **"Lung-Cancer-Detection"** repository

### 3. Configure Service
Use these exact settings:

**Basic Settings:**
- **Name:** `lung-cancer-detection-ai` (or any name you prefer)
- **Branch:** `main`
- **Runtime:** `Python 3`

**Build & Deploy:**
- **Build Command:** `pip install -r requirements_streamlit.txt`
- **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

**Advanced Settings:**
- **Environment:** `Python 3.11` (recommended)
- **Instance Type:** `Free` (default)

### 4. Add Database (Optional)
For full functionality with data persistence:

- Click **"New +"** → **"PostgreSQL"**
- **Name:** `lung-cancer-db`
- **Plan:** `Free` 
- **User:** Leave default
- Click **"Create Database"**

### 5. Connect Database to Web Service
- Go back to your web service settings
- Click **"Environment"** tab
- Add environment variable:
  - **Key:** `DATABASE_URL`
  - **Value:** Select your PostgreSQL database from dropdown

### 6. Deploy!
- Click **"Create Web Service"**
- Wait 5-10 minutes for deployment
- Your app will be live at: `https://lung-cancer-detection-ai.onrender.com`

## What You'll Get

✅ **Professional AI application** running 24/7  
✅ **Your own custom URL** to share  
✅ **Automatic updates** when you push to GitHub  
✅ **Free database** for storing results  
✅ **Mobile-responsive** interface  

## Troubleshooting

**Build fails?**
- Check that `requirements_streamlit.txt` exists in your repository
- Ensure the build command is exactly: `pip install -r requirements_streamlit.txt`

**App won't start?**
- Verify start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- Check that `app.py` exists in your repository root

**Database connection issues?**
- Ensure `DATABASE_URL` environment variable is set
- Database connection is optional - app works without it

## Custom Domain (Optional)

Once deployed, you can add a custom domain:
1. Buy a domain (GoDaddy, Namecheap, etc.)
2. In Render dashboard → Settings → Custom Domain
3. Add your domain and configure DNS

Your AI application is now live and accessible worldwide! 🎉