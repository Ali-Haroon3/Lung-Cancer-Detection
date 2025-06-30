# Git Integration Troubleshooting

## Common Issues with Replit Git Integration

### Issue 1: Commits Not Pushing
**Symptoms**: Changes are committed but don't appear on GitHub
**Solutions**:
1. Check if the remote origin is correctly set to your repository
2. Verify you have push permissions to the repository
3. Ensure your GitHub token/credentials are valid

### Issue 2: Authentication Problems
**Symptoms**: "Permission denied" or authentication errors
**Solutions**:
1. Go to Replit Secrets and add `GITHUB_TOKEN` with your Personal Access Token
2. Generate a new token at GitHub → Settings → Developer settings → Personal access tokens
3. Token needs `repo` permissions

### Issue 3: Files Not Being Added
**Symptoms**: Some files missing from commits
**Solutions**:
1. Check .gitignore file - ensure important files aren't excluded
2. Large files (>100MB) may be rejected by GitHub
3. Binary files like models (best_model.h5) might need Git LFS

## Manual Push from Replit Shell

If the Git integration fails, use these commands in the Replit Shell:

```bash
# Check current status
git remote -v
git status

# Force add and commit
git add .
git commit -m "Update: $(date)"

# Push with force if needed
git push origin main --force
```

## Alternative: Export and Manual Upload

1. In Replit, click the three dots menu → "Download as zip"
2. Extract files to your local lung-cancer-detection folder
3. Use local Git commands:
```bash
git add .
git commit -m "Replit project sync"
git push origin main
```

## Files Currently Ready for Sync

- Complete Streamlit application (app.py + pages/)
- CNN model implementations (models/)
- Database schema and services (database/)
- Utility modules (utils/)
- Sample data generation (sample_data/)
- Configuration files (.gitignore, README.md)
- Documentation (GITHUB_SETUP.md, this file)

## Verification

After successful push, your GitHub repository should contain:
- All Python source files
- Database schema and models
- Complete project documentation
- Proper .gitignore excluding replit.md