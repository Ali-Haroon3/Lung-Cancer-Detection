# Vercel Deployment Guide with Custom Domain

## Overview
This guide walks you through deploying your lung cancer detection AI application on Vercel with a custom domain.

## Prerequisites
- GitHub repository with your application code
- Vercel account (free tier available)
- Custom domain (purchased from any domain registrar)

## Step 1: Prepare Repository for Vercel

### 1.1 Create Vercel Configuration
Your `vercel.json` is already configured with:
- Streamlit entry point: `streamlit_app.py`
- Build and start commands
- Environment variable support

### 1.2 Entry Point File
Your `streamlit_app.py` serves as the Vercel entry point, importing your main application.

### 1.3 Dependencies
Your `requirements_vercel.txt` contains optimized dependencies for deployment.

## Step 2: Deploy to Vercel

### 2.1 Connect Repository
1. Go to [vercel.com](https://vercel.com) and sign up/login
2. Click "New Project"
3. Import your GitHub repository
4. Vercel will auto-detect it as a Python project

### 2.2 Configure Environment Variables
In Vercel dashboard, add these environment variables:
```
DATABASE_URL=your_postgresql_connection_string
GITHUB_TOKEN=your_github_token (if needed)
```

### 2.3 Deploy
- Click "Deploy"
- Vercel will build and deploy your application
- You'll get a `.vercel.app` domain initially

## Step 3: Add Custom Domain

### 3.1 Purchase Domain
Buy a domain from any registrar:
- Namecheap, GoDaddy, Google Domains, Cloudflare, etc.
- Example domains: `lungcancer-ai.com`, `medicalai-detector.com`

### 3.2 Configure Domain in Vercel
1. In your Vercel project dashboard
2. Go to "Settings" → "Domains"
3. Click "Add Domain"
4. Enter your custom domain (e.g., `yourdomain.com`)

### 3.3 DNS Configuration
Vercel will provide DNS records to add to your domain registrar:

**Option A: Use Vercel DNS (Recommended)**
- Point your domain's nameservers to Vercel
- Vercel manages all DNS automatically

**Option B: Configure DNS manually**
Add these records at your domain registrar:
```
Type: A
Name: @ (or your domain)
Value: 76.76.19.61

Type: CNAME  
Name: www
Value: cname.vercel-dns.com
```

### 3.4 SSL Certificate
- Vercel automatically provides SSL certificates
- Your site will be available at `https://yourdomain.com`

## Step 4: Production Optimizations

### 4.1 Database Configuration
Ensure your PostgreSQL database is production-ready:
- Use a managed service (Heroku Postgres, AWS RDS, Supabase)
- Configure connection pooling
- Set proper security rules

### 4.2 Model File Handling
With Git LFS configured:
- Large model files deploy automatically
- No size limitations on Vercel
- Fast loading from CDN

### 4.3 Performance Monitoring
- Enable Vercel Analytics
- Monitor function execution times
- Set up error tracking

## Step 5: Custom Domain Examples

Popular domain patterns for medical AI:
- `[yourname]-medicalai.com`
- `lungcancer-detector.com`
- `ai-radiology-tools.com`
- `medical-imaging-ai.net`

## Step 6: Post-Deployment Checklist

✅ Application loads correctly on custom domain
✅ Database connections work
✅ Model predictions function properly
✅ SSL certificate is active (https://)
✅ All pages navigate correctly
✅ File uploads work
✅ Model training completes successfully

## Troubleshooting

### Build Failures
- Check `requirements_vercel.txt` dependencies
- Verify Python version compatibility
- Review build logs in Vercel dashboard

### Database Connection Issues
- Verify DATABASE_URL environment variable
- Check PostgreSQL server accessibility
- Ensure connection string format is correct

### Domain Configuration Issues
- Allow 24-48 hours for DNS propagation
- Verify DNS records are correct
- Check domain registrar settings

## Cost Considerations

### Vercel Pricing
- **Hobby (Free)**: Perfect for personal projects
  - 100 GB bandwidth/month
  - Unlimited personal repositories
  - Custom domains included

- **Pro ($20/month)**: For production applications
  - 1 TB bandwidth/month
  - Team collaboration features
  - Advanced analytics

### Domain Costs
- `.com` domains: $10-15/year typically
- Consider `.ai` for AI applications: $60-80/year
- `.health` or `.care` for medical focus: $40-60/year

## Security Best Practices

- Use environment variables for sensitive data
- Enable Vercel's security headers
- Configure proper CORS settings
- Use HTTPS everywhere (automatic with Vercel)
- Regular dependency updates

Your application will be live at your custom domain within minutes of completing these steps!