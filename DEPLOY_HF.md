# Deploy to Hugging Face Spaces

This app is a **Streamlit + TensorFlow** app. It needs a persistent server with
~1 GB+ RAM, so it **cannot run on Vercel** (serverless: size/memory/time caps,
no persistent process). Hugging Face Spaces' free tier gives 16 GB RAM and keeps
the process warm, so the model loads once and stays loaded.

## One-time setup

### 1. Create the Space
Go to <https://huggingface.co/new-space> → **SDK: Streamlit**, pick a name
(e.g. `lung-cancer-detection`), Hardware: **CPU basic (free)**. It starts empty.

### 2. Push the code (model auto-downloads, so no Git-LFS needed)
```bash
git clone https://huggingface.co/spaces/<your-hf-username>/lung-cancer-detection hf-space
cd hf-space
git remote add gh https://github.com/Ali-Haroon3/Lung-Cancer-Detection.git
git fetch gh main
git checkout gh/main -- .                 # bring the app files into the Space
git reset                                  # unstage
rm -f models/lung_cancer_model.keras       # ~56 MB; the app downloads it on first run
git add -A
git commit -m "Deploy lung cancer detection app"
git push                                    # use an HF access token (write) as the password
```

### 3. Done
The Space builds (installs `requirements.txt` + `packages.txt`), runs `app.py`,
and on the first analyze it downloads the model from `MODEL_URL` (defaults to the
GitHub raw URL) and caches it for the life of the container.

## Notes
- **Model source**: set the `MODEL_URL` env var in the Space settings to override
  where the model is fetched from. Or commit the model into the Space with Git LFS
  (`git lfs track "*.keras"`) to skip the runtime download entirely.
- **Database**: the optional Postgres analytics pages need `DATABASE_URL`. Without
  it the app still runs — the home/analyze flow works; those pages just show a
  "database not configured" notice.
- **Why not Vercel**: TensorFlow's wheel alone exceeds Vercel's serverless size
  limit, and Streamlit needs a long-lived process. That mismatch is what caused
  the "loads forever", usage burn, and crashes.
