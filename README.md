# DSCI-560 Lab 5: Reddit Scraping, Preprocessing, Clustering & Automation

This project uses **a virtual environment** for dependencies. Set it up once, then run the pipeline and scripts from that env.

---

## 1. Create and use a virtual environment

### Windows (PowerShell or Command Prompt)

```powershell
# From the project folder (dsci560_lab5)
cd path\to\dsci560_lab5

# Create the virtual environment (uses your default Python)
python -m venv venv

# Activate it
.\venv\Scripts\activate

# You should see (venv) in your prompt. Install dependencies:
pip install -r requirements.txt

# Download NLTK data (required once per env)
python -c "import nltk; nltk.download('punkt')"
```

### Linux (Ubuntu / WSL)

```bash
cd /path/to/dsci560_lab5

# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

---

## 2. Run the project (with venv active)

Make sure the venv is activated (`(venv)` in the prompt, or `source venv/bin/activate` on Linux).

### One-off: OCR → Embedding → Clustering (from CSV)

Uses `reddit_posts.csv` and optional `data/processed_data/reddit_ocr_results.json`:

```bash
python data_processing_and_analysis.py
```

### Automation: periodic scrape + process + search

```bash
# Update every 5 minutes (fetch → process → save); optional: run search in another terminal
python run_pipeline.py 5

# Search-only: type keywords to find closest cluster and see messages + graph
python run_pipeline.py
```

### Scrape only (optional)

```bash
python reddit_scraper.py
# Edit subreddits/count in the script if needed; outputs reddit_posts.csv
```

---

## 3. Optional: Tesseract (OCR)

For text extraction from images in posts:

- **Windows:** Install from [UB-Mannheim tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Default path `C:\Program Files\Tesseract-OCR\tesseract.exe` is detected automatically.
- **Linux:** `sudo apt install tesseract-ocr`

If Tesseract is not installed, OCR is skipped and the rest of the pipeline still runs.

---

## 4. Project layout (main files)

| File / folder           | Purpose |
|-------------------------|--------|
| `venv/`                 | Virtual environment (create with `python -m venv venv`); do not commit. |
| `requirements.txt`      | Pip dependencies; install inside venv. |
| `reddit_scraper.py`     | Fetch Reddit posts (old Reddit JSON API), save to CSV. |
| `data_processing_and_analysis.py` | OCR (optional), embedding (Tfidf+PCA), cluster tree, save vectors/labels. |
| `run_pipeline.py`      | Periodic fetch + process + storage; or search by keywords. |
| `reddit_posts.csv`      | Fetched posts (created/updated by scraper or pipeline). |
| `data/processed_data/`  | OCR JSON, doc vectors, cluster labels, tree images. |

---
