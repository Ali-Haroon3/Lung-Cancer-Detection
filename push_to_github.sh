#!/bin/bash

echo "🚀 Setting up Git with your Ali-Haroon3 account credentials..."

# Configure Git with your personal GitHub account
git config user.name "Ali-Haroon3"
git config user.email "your.email@gmail.com"

echo "📧 Please update the email in this script with your actual GitHub email"
echo "🚀 Pushing cleaned Lung Cancer Detection AI to Ali-Haroon3 GitHub..."

# Add all the cleaned files
git add .

# Commit with a meaningful message
git commit -m "Clean up project: Remove unnecessary files and prepare for deployment

- Removed all Replit-specific documentation files
- Cleaned up cached Python files and temporary directories  
- Updated README with proper Ali-Haroon3 GitHub links
- Added deployment configurations for Render hosting
- Project now ready for professional GitHub hosting"

# Push to your Ali-Haroon3 repository
git push origin main

echo "✅ Successfully pushed to https://github.com/Ali-Haroon3/Lung-Cancer-Detection"
echo "🌐 Your repository is now updated with the clean project structure!"