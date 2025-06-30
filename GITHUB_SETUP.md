# GitHub Integration Setup Guide

This guide helps you connect your lung cancer detection project to GitHub for automatic version control.

## Step 1: Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" button in the top right corner
3. Select "New repository"
4. Repository settings:
   - **Name**: `lung-cancer-detection-cnn` (or your preferred name)
   - **Description**: "Deep learning application for lung cancer detection using CNN and medical imaging"
   - **Visibility**: Choose Public or Private
   - **Important**: Do NOT initialize with README, .gitignore, or license (we already have files)

## Step 2: Get Your Repository URL

After creating the repository, copy the HTTPS URL that looks like:
```
https://github.com/YOUR_USERNAME/lung-cancer-detection-cnn.git
```

## Step 3: Terminal Commands Setup

Open your Replit Shell (Tools → Shell) and run these commands:

### Configure Git Identity
```bash
git config --global user.name "Your Full Name"
git config --global user.email "your-email@example.com"
```

### Add Files and Create Initial Commit
```bash
# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit: Lung Cancer Detection CNN Application

- Complete Streamlit web application for medical imaging
- CNN model with ResNet50, DenseNet121, EfficientNetB0 architectures
- PostgreSQL database integration
- Data upload, model training, evaluation, and prediction features
- Real-time training progress monitoring
- Medical-specific evaluation metrics"
```

### Connect to Your GitHub Repository
```bash
# Add your GitHub repository as remote origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

## Step 4: Verify Setup

After pushing, check your GitHub repository to confirm all files are uploaded.

## Step 5: Future Commits

### Manual Commits
Whenever you want to save changes to GitHub:
```bash
git add .
git commit -m "Describe your changes here"
git push origin main
```

### Using the Auto-Commit Script
Run the provided script for quick commits:
```bash
./auto_commit.sh
```

## Project Structure on GitHub

Your repository will include:
```
├── app.py                    # Main Streamlit application
├── pages/                    # Multi-page app structure
│   ├── 1_Data_Upload.py
│   ├── 2_Model_Training.py
│   ├── 3_Model_Evaluation.py
│   ├── 4_Prediction.py
│   └── 5_Database_Management.py
├── models/                   # CNN model implementations
├── database/                 # Database management
├── utils/                    # Utility functions
├── sample_data/             # Sample medical images
├── pyproject.toml           # Python dependencies
├── replit.md                # Project documentation
├── .gitignore               # Git ignore rules
└── auto_commit.sh           # Auto-commit script
```

## Best Practices

1. **Commit frequently** with descriptive messages
2. **Use meaningful commit messages** that describe what changed
3. **Keep sensitive data out** - the .gitignore file prevents accidental commits of:
   - Database files
   - Model files (*.h5)
   - API keys and secrets
   - Large data files

## Troubleshooting

### If you get authentication errors:
- GitHub may require a Personal Access Token instead of password
- Go to GitHub Settings → Developer settings → Personal access tokens
- Create a token with repo permissions
- Use the token as your password when prompted

### If you get "repository already exists" error:
- The remote is already set up
- Skip step 3 and just use: `git push origin main`

### If you get merge conflicts:
- Run: `git pull origin main` first
- Resolve any conflicts
- Then push your changes

## Automatic Commits

The `auto_commit.sh` script provides one-command commits with timestamps for quick development cycles.