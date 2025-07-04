# Custom Domain Deployment Guide

## Quick Start for Custom Domain Deployment

### 1. Commit Deployment Files
```bash
git add vercel.json streamlit_app.py requirements_vercel.txt VERCEL_DEPLOYMENT.md
git commit -m "Add Vercel deployment configuration for custom domain

- Configure vercel.json with Streamlit entry point
- Add streamlit_app.py as Vercel entry point
- Optimize requirements_vercel.txt for production
- Include comprehensive deployment guide"

git push origin main
```

### 2. Deploy to Vercel

#### Option A: Connect via GitHub (Recommended)
1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Vercel auto-detects the configuration
5. Click "Deploy"

#### Option B: Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from repository
vercel --prod
```

### 3. Add Your Custom Domain

#### Purchase a Domain
Popular options for medical AI:
- `yourname-medicalai.com`
- `lungcancer-detector.com` 
- `ai-diagnostic-tools.com`
- `medical-imaging-ai.net`

#### Configure Domain in Vercel
1. In Vercel project dashboard
2. Go to "Settings" → "Domains"
3. Click "Add Domain"
4. Enter your domain: `yourdomain.com`

#### DNS Setup
Vercel provides these options:

**Option A: Use Vercel Nameservers (Easiest)**
- Point your domain to Vercel's nameservers
- Vercel handles all DNS automatically

**Option B: Manual DNS Records**
Add at your domain registrar:
```
A Record: @ → 76.76.19.61
CNAME: www → cname.vercel-dns.com
```

### 4. Environment Variables
In Vercel dashboard, add:
```
DATABASE_URL=your_postgresql_connection_string
GITHUB_TOKEN=your_github_token
```

### 5. SSL & Security
- SSL certificate automatically provided
- Site accessible at `https://yourdomain.com`
- Security headers configured in vercel.json

## Production Checklist

✅ Repository deployed to Vercel
✅ Custom domain configured
✅ SSL certificate active
✅ Database connection working
✅ Model files loading via Git LFS
✅ All application features functional

## Domain Suggestions by Category

### Professional Medical AI
- medicalai-diagnostics.com
- precision-radiology.com
- clinical-ai-tools.com

### Lung Cancer Specific
- lungcancer-detection.com
- thoracic-ai-scanner.com
- pulmonary-diagnostics.ai

### Personal Branding
- [yourname]-medicalai.com
- dr[yourname]-aitools.com
- [yourname]-diagnostics.com

## Cost Breakdown

### Vercel (Free Tier)
- ✅ Custom domains included
- ✅ SSL certificates
- ✅ 100GB bandwidth/month
- ✅ Unlimited personal projects

### Domain Registration
- `.com`: $10-15/year
- `.ai`: $60-80/year  
- `.health`: $40-60/year

### Total Annual Cost
Starting from $10/year for professional medical AI deployment!

Your application will be live at your custom domain within 10 minutes of completing these steps.