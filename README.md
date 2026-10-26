# CropYield AI — FAOSTAT Crop Yield Prediction

End-to-end Statistical Interface Lab project: FAOSTAT page scraping, data collection, preprocessing, exploratory analysis, regression training, and Streamlit GUI.

## Important design choice
This first version predicts FAOSTAT's recorded crop yield from **Country/Area, Crop, Year, and Area harvested**. It intentionally does not use production as an input feature because yield is derived from production / area. FAOSTAT is country/area-level data, so the output is a historical-statistics-based estimate, not a farm-level agronomic forecast.

## Project structure
- `notebooks/CropYield_AI_Colab.ipynb`: run the complete workflow in Google Colab
- `src/faostat_scraper.py`: scrape FAOSTAT page links and download/read its normalized data archive
- `src/train_model.py`: prepare data, train models, save metrics and model
- `app.py`: Streamlit website
- `models/`: receives trained model artifacts after training
- `data/`: receives raw and cleaned data after collection/training

## Run in Google Colab
1. Upload `CropYield_AI_Colab.ipynb` or open it from this repository in Colab.
2. Run all cells in order. The first download can take a few minutes because the official normalized dataset is large.
3. Download the generated `models/` and `data/crop_yield_training.csv` artifacts, or run the notebook's ZIP export cell.
4. Add the generated artifacts to this repository before deploying the website.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
# Optional: refresh the official raw FAOSTAT archive first.
python src/faostat_scraper.py
# Rebuilds a local-compatible model from the raw archive or existing Colab CSV.
python src/train_model.py
streamlit run app.py
```

The supplied `run_local_windows.bat` starts the app from the saved artifacts; it does
not re-download FAOSTAT on every launch. If a model was serialized with a different
scikit-learn version, run `python src/train_model.py` once to rebuild it locally.

The scraper writes the complete raw dataset to `data/raw/faostat_qcl_raw.csv` when run via the notebook workflow. The CLI scraper currently prints a sample by default; use the notebook for the full pipeline.

## Deploy
Push the repository to GitHub. Ensure `models/best_model.joblib`, `models/model_metadata.json`, `models/model_metrics.csv`, and `data/crop_yield_training.csv` are present. Then deploy `app.py` at https://share.streamlit.io/ using `requirements.txt`.

## Web scraping and data provenance
The script requests the public FAOSTAT QCL page with `requests`, parses anchors using BeautifulSoup, and looks for the official normalized ZIP archive link. FAOSTAT may render parts of the page with JavaScript; if no link is exposed in the returned HTML, the script uses the documented public bulk-download URL as a fallback. The ZIP contains normalized CSV data. Record the retrieval date and any changes to the source URL in your lab report.

## Evaluation cautions
The starter script uses a reproducible random train/test split as a baseline. Because observations are grouped by country/crop/year, this may produce optimistic metrics. For a stronger final evaluation, hold out later years or hold out entire country/crop groups, then compare with a simple mean/last-year baseline. Always report the split method.
