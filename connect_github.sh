#!/bin/bash

# Quick setup script to connect to your lung-cancer-detection repository
# Replace YOUR_USERNAME with your actual GitHub username

echo "Connecting to your lung-cancer-detection repository..."

# Set up Git configuration (replace with your details)
read -p "Enter your GitHub username: " USERNAME
read -p "Enter your email: " EMAIL

git config --global user.name "$USERNAME"
git config --global user.email "$EMAIL"

# Add files and create initial commit
git add .
git commit -m "Initial commit: Lung Cancer Detection CNN Application

Complete Streamlit web application featuring:
- CNN models with ResNet50, DenseNet121, EfficientNetB0
- PostgreSQL database integration  
- Real-time training monitoring
- Medical evaluation metrics
- Prediction interface with confidence scoring"

# Connect to your repository
git remote add origin https://github.com/$USERNAME/lung-cancer-detection.git

# Push to GitHub
git push -u origin main

echo "Successfully connected to GitHub repository!"
echo "Use ./auto_commit.sh for future updates"