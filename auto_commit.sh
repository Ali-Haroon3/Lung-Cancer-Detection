#!/bin/bash

# Auto-commit script for lung cancer detection project
# Run this script whenever you want to save changes to GitHub

echo "Auto-committing changes to GitHub..."

# Add all changes (excluding replit.md)
git add .

# Create commit with timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
git commit -m "Auto-update: $TIMESTAMP"

# Push to GitHub
git push origin main

echo "Changes committed and pushed to GitHub successfully!"