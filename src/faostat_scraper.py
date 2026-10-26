"""
FAOSTAT data collection + web-scraping utilities.
This script scrapes the FAOSTAT Crops and livestock products page for links,
then downloads the official normalized bulk CSV archive. It does not scrape
or overload individual result pages.
"""
from pathlib import Path
from urllib.parse import urljoin
import io
import re
import zipfile
import requests
import pandas as pd
from bs4 import BeautifulSoup

FAOSTAT_PAGE = "https://www.fao.org/faostat/en/#data/QCL"
# Official FAOSTAT bulk download for Production: Crops and livestock products,
# normalized long format. Fallback is used if FAOSTAT's page is JS-rendered.
FALLBACK_URL = (
    "https://bulks-faostat.fao.org/production/"
    "Production_Crops_Livestock_E_All_Data_(Normalized).zip"
)
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "CropYieldAILabProject/1.0 (educational research; contact: student project)"
}

def scrape_faostat_download_links(page_url=FAOSTAT_PAGE):
    """Scrape links from the public FAOSTAT QCL page; return matching URLs."""
    response = requests.get(page_url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    found = []
    for tag in soup.find_all(["a", "link"]):
        href = tag.get("href", "")
        label = " ".join([
            tag.get_text(" ", strip=True),
            tag.get("title", ""),
            tag.get("aria-label", "")
        ]).lower()
        full_url = urljoin(page_url, href)
        if ("production_crops_livestock" in full_url.lower()
                or ("bulk" in label and "crop" in label)):
            found.append(full_url)
    # JS-rendered pages may not expose the link in the HTML response.
    if not found:
        found = [FALLBACK_URL]
    # Keep unique URLs, preserving order.
    return list(dict.fromkeys(found))

def download_faostat_archive(url=None, output_path=RAW_DIR / "faostat_qcl_normalized.zip"):
    """Download the official FAOSTAT normalized archive."""
    if url is None:
        candidates = scrape_faostat_download_links()
        url = next((u for u in candidates if u.lower().endswith(".zip")), FALLBACK_URL)
    with requests.get(url, headers=HEADERS, stream=True, timeout=(30, 180)) as r:
        r.raise_for_status()
        total = 0
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    if not zipfile.is_zipfile(output_path):
        Path(output_path).unlink(missing_ok=True)
        raise ValueError("The downloaded response was not a ZIP archive. Check the FAOSTAT URL.")
    print(f"Downloaded {total / (1024*1024):.1f} MB to {output_path}")
    return Path(output_path)

def read_faostat_archive(zip_path):
    """Read the main CSV in the ZIP without extracting the full archive."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV file found in FAOSTAT ZIP archive.")
        # Prefer the largest CSV (the primary data file).
        csv_name = max(csv_names, key=lambda n: zf.getinfo(n).file_size)
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False, encoding_errors="replace")
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Read {len(df):,} rows from {csv_name}")
    return df

def collect_faostat_data():
    print("Scraping FAOSTAT page for dataset links...")
    links = scrape_faostat_download_links()
    print("Candidate link(s):")
    for link in links:
        print(" -", link)
    zip_path = download_faostat_archive()
    df = read_faostat_archive(zip_path)
    return df

if __name__ == "__main__":
    data = collect_faostat_data()
    output = DATA_DIR / "raw" / "faostat_qcl_raw.csv"
    data.to_csv(output, index=False)
    print(data.head())
    print(f"Saved full raw dataset to {output}")
