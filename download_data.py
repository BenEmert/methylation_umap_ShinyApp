# download_data.py
import os
import requests
import GEOparse
import logging\
from pathlib import Path
import shutil

# Metadata (Hugging Face / Local)
META_URL = "https://huggingface.co/datasets/bemert/GSE140686_GPL13534/resolve/main/41467_2020_20603_MOESM4_ESM.xlsx"
META_FILENAME = "41467_2020_20603_MOESM4_ESM.xlsx"

# GEO Data Configuration
GEO_ACCESSION = "GSE140686"
# This string must exist in the supplementary file URL you want to target
# (e.g. "matrix_processed" or the full filename "GSE140686_GPL13534_matrix_processed.txt.gz")
GEO_FILE_PATTERN = "GSE140686_GPL13534_matrix_processed.txt.gz" 
DATA_DIR = Path('./data/raw')
# =================================================

# Silence GEOparse logs
logging.getLogger('GEOparse').setLevel(logging.ERROR)

def download_file(url, outfile):
    """Generic function to download a file from a URL."""
    if outfile.exists():
        print(f"   [SKIP] Found existing file: {outfile.name}")
        return

    print(f"   [DOWNLOADING] {outfile.name}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(outfile, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"   [SUCCESS] Saved to {outfile}")
    except Exception as e:
        print(f"   [ERROR] Failed to download {url}. Reason: {e}")
        raise e

def download_geo_supp_file(geo_accession, file_pattern, output_dir):
    """
    Fetches the GEO Series metadata and downloads the specific supplementary file
    matching the provided pattern.
    """
    print(f"   [GEO] Querying {geo_accession}...")
    
    # 1. Get GEO Object (metadata only, how='quick')
    try:
        gse = GEOparse.get_GEO(geo_accession, destdir="./", how="quick", silent=True)
    except Exception as e:
        print(f"   [ERROR] Could not fetch GEO metadata: {e}")
        raise e

    # 2. Find the correct URL from metadata
    found_url = None
    for url in gse.metadata.get('supplementary_file', []):
        if file_pattern in url:
            found_url = url
            break
            
    if not found_url:
        raise ValueError(f"Could not find a supplementary file containing '{file_pattern}' in {geo_accession}")

    filename = found_url.split('/')[-1]
    outfile = output_dir / filename
    
    print(f"   [GEO] Found target: {filename}")
    
    download_url_link = found_url.replace("ftp://", "https://")
    
    download_file(download_url_link, outfile)


def main():
    print("=== Phase 1: Data Acquisition ===")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"1. Target directory: {DATA_DIR}")

    repo_root_meta = Path("41467_2020_20603_MOESM4_ESM.xlsx")
    target_meta = DATA_DIR / META_FILENAME
    
    if repo_root_meta.exists():
        print("   [INFO] Found metadata in repository root. Moving to data folder.")
        shutil.copy(repo_root_meta, target_meta)
    else:
        download_file(META_URL, target_meta)

    try:
        download_geo_supp_file(GEO_ACCESSION, GEO_FILE_PATTERN, DATA_DIR)
    except Exception as e:
        print(f"ERROR downloading GEO data: {e}")

    print("=== Data Download Complete ===")

if __name__ == "__main__":
    main()